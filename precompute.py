"""
Precomputation for the score test.

Regresses each gene's counts onto the cell covariates and produces the score-test
pieces (a, w, D) consumed by the kernel. Following Barry et al., the negative-binomial
GLM is fit quickly by (i) a Poisson GLM, (ii) estimating the NB size parameter theta
from the Poisson residuals, and (iii) taking the NB coefficients equal to the Poisson
coefficients. This makes the pipeline self-contained: raw counts -> a,w,D -> kernel
-> p-values.

Two interchangeable implementations, selected by device:
  * a per-gene NumPy/SciPy reference (`precompute_gene_batch`), and
  * a batched PyTorch path (`precompute_gene_batch_torch`) that fits all genes at
    once as tensor operations and runs on CPU or CUDA in float64.
The dispersion (theta) fit dominates the cost, and it is dominated by digamma/trigamma
evaluations; batching those across genes is what makes the GPU path fast. The two
implementations agree to floating-point tolerance (see test_gpu_sceptre.py).

The outputs (a, w, D) feed gpu_sceptre.run_association_test directly.
"""
import numpy as np
from scipy.special import digamma, polygamma   # trigamma(x) = polygamma(1, x)

THETA_LO, THETA_HI = 0.01, 1000.0


# ---- Poisson GLM via iteratively reweighted least squares -------------------
def poisson_irls(y, X, max_iter=50, tol=1e-8):
    """Fit log-link Poisson GLM by iteratively reweighted least squares.
    Returns (coefs (d,), fitted mean mu (n,)). Initialised at mu = y + 0.1."""
    mu = y + 0.1
    eta = np.log(mu)
    beta = np.zeros(X.shape[1])
    for _ in range(max_iter):
        mu = np.exp(eta)
        mu = np.maximum(mu, 1e-10)
        W = mu                                   # Poisson IRLS weight
        z = eta + (y - mu) / mu                  # working response
        XtW = X.T * W                            # (d,n)
        beta_new = np.linalg.solve(XtW @ X, XtW @ z)
        eta_new = X @ beta_new
        if np.max(np.abs(beta_new - beta)) < tol:
            beta, eta = beta_new, eta_new
            break
        beta, eta = beta_new, eta_new
    return beta, np.exp(eta)


# ---- NB size parameter theta (Newton MLE, method-of-moments fallback) -------
def _nb_score(th, mu, y):
    return np.sum(digamma(th + y) - digamma(th) + np.log(th) + 1
                  - np.log(th + mu) - (y + th) / (mu + th))

def _nb_info(th, mu, y):
    return np.sum(polygamma(1, th) - polygamma(1, th + y) - 1.0 / th
                  + 2.0 / (mu + th) - (y + th) / ((mu + th) ** 2))

def _theta_mle(t0, y, mu, limit=50, eps=None):
    eps = eps or np.finfo(float).eps ** 0.25
    it, delta = 0, 1.0
    with np.errstate(all="ignore"):
        while it < limit and abs(delta) > eps:
            it += 1
            t0 = min(abs(t0), 1e8)
            info = _nb_info(t0, mu, y)
            if info == 0 or not np.isfinite(info):
                break
            delta = _nb_score(t0, mu, y) / info
            t0 += delta
            if not np.isfinite(t0):
                break
    warn = (t0 < 0) or (it == limit) or (not np.isfinite(t0))
    return t0, warn

def _theta_mm(t0, y, mu, dfr, limit=50, eps=None):
    eps = eps or np.finfo(float).eps ** 0.25
    it, delta = 0, 1.0
    with np.errstate(all="ignore"):
        while it < limit and abs(delta) > eps:
            it += 1
            t0 = min(abs(t0), 1e8)
            den = np.sum((y - mu) ** 2 / (mu + t0) ** 2)
            if den == 0 or not np.isfinite(den):
                break
            num = np.sum((y - mu) ** 2 / (mu + mu ** 2 / t0)) - dfr
            delta = num / den
            t0 -= delta
            if not np.isfinite(t0):
                break
    warn = (t0 < 0) or (it == limit) or (not np.isfinite(t0))
    return t0, warn

def estimate_theta(y, mu, dfr):
    """MLE, falling back to method-of-moments, then to the pilot estimate."""
    t0 = len(y) / np.sum((y / mu - 1.0) ** 2)     # pilot
    try:
        est, warn = _theta_mle(t0, y, mu)
        if warn:
            est, warn = _theta_mm(t0, y, mu, dfr)
            if warn:
                est = t0
    except Exception:
        est = t0
    return est


# ---- Score-test pieces a, w, D ---------------------------------------------
def compute_D_matrix(ZtwZ, wZ):
    """D = Lambda^{-1/2} Uᵀ (wZ)ᵀ, from eigendecomp ZᵀWZ = U Λ Uᵀ. Shape (d, n).
    DᵀD = WZ (ZᵀWZ)⁻¹ (WZ)ᵀ (the score-test variance quadratic form)."""
    vals, U = np.linalg.eigh(ZtwZ)
    return (U / np.sqrt(vals)).T @ wZ.T           # (d,n): (Λ^{-1/2} Uᵀ)(wZ)ᵀ

def precompute_pieces(y, X, coefs, theta):
    """Returns a, w (n,) and D (d,n) for one gene."""
    mu = np.exp(X @ coefs)
    denom = 1.0 + mu / theta
    w = mu / denom
    a = (y - mu) / denom
    wZ = w[:, None] * X                           # (n,d)
    ZtwZ = X.T @ wZ                               # (d,d)
    D = compute_D_matrix(ZtwZ, wZ)                # (d,n)
    return a, w, D, mu


def response_precomputation(y, X):
    """Per-gene: Poisson fit -> theta (clamped) -> (coefs, theta)."""
    coefs, mu_pois = poisson_irls(y, X)
    dfr = len(y) - X.shape[1]
    theta = float(np.clip(estimate_theta(y, mu_pois, dfr), THETA_LO, THETA_HI))
    return coefs, theta


def precompute_gene_batch(Y, X):
    """Y (G,n) counts, X (n,d) covariates (incl. intercept). Returns A,W,D,MU.
    A,W (G,n); D (G,d,n); MU (G,n) NB fitted mean. Per-gene NumPy reference."""
    G, n = Y.shape
    d = X.shape[1]
    A = np.empty((G, n)); W = np.empty((G, n)); D = np.empty((G, d, n)); MU = np.empty((G, n))
    for g in range(G):
        coefs, theta = response_precomputation(Y[g], X)
        a, w, Dg, mu = precompute_pieces(Y[g], X, coefs, theta)
        A[g], W[g], D[g], MU[g] = a, w, Dg, mu
    return A, W, D, MU


# ---- Batched PyTorch precompute (all genes at once; CPU or CUDA, float64) ----
def _poisson_irls_batched(Y, X, max_iter=50, tol=1e-8):
    """Poisson IRLS for a whole gene batch. Y (G,n), X (n,d) -> fitted mean (G,n).
    Normal equations are formed and solved for all genes simultaneously."""
    import torch
    G, d = Y.shape[0], X.shape[1]
    eta = torch.log(Y + 0.1)
    beta = torch.zeros(G, d, dtype=Y.dtype, device=Y.device)
    for _ in range(max_iter):
        mu = torch.exp(eta).clamp_min(1e-10)
        z = eta + (Y - mu) / mu                                # working response
        XtWX = torch.einsum('ni,gn,nj->gij', X, mu, X)         # (G,d,d)
        XtWz = torch.einsum('ni,gn->gi', X, mu * z)            # (G,d)
        beta_new = torch.linalg.solve(XtWX, XtWz.unsqueeze(-1)).squeeze(-1)
        eta = beta_new @ X.T
        if (beta_new - beta).abs().max() < tol:
            beta = beta_new
            break
        beta = beta_new
    return torch.exp(eta)


def _theta_batched(Y, mu, dfr, limit=50):
    """NB size parameter theta for a whole gene batch (G,). Newton MLE with a
    method-of-moments and pilot fallback, mirroring estimate_theta but vectorized
    across genes: converged/diverged genes are frozen by a per-gene mask while the
    remaining genes keep iterating. Returns theta (G,), clamped to [THETA_LO,THETA_HI]."""
    import torch
    G, n = Y.shape
    eps = float(np.finfo(np.float64).eps ** 0.25)
    digamma_t = torch.special.digamma
    trigamma_t = lambda x: torch.special.polygamma(1, x)
    t0 = n / ((Y / mu - 1.0) ** 2).sum(1)                      # pilot (G,)

    def newton(step, init):
        t = init.clone()
        active = torch.ones(G, dtype=torch.bool, device=Y.device)
        it_count = torch.zeros(G, device=Y.device)
        for _ in range(limit):
            if not active.any():
                break
            t = torch.minimum(t.abs(), t.new_tensor(1e8))
            delta, bad = step(t)
            new_t = t + delta
            broke = bad | ~torch.isfinite(new_t)
            t = torch.where(active & ~broke, new_t, t)
            it_count += active.double()
            active = active & ~broke & (delta.abs() > eps)
        warn = (t < 0) | (it_count >= limit) | ~torch.isfinite(t)
        return t, warn

    def mle_step(t):
        tc = t[:, None]
        info = (trigamma_t(tc) - trigamma_t(tc + Y) - 1.0 / tc
                + 2.0 / (mu + tc) - (Y + tc) / ((mu + tc) ** 2)).sum(1)
        score = (digamma_t(tc + Y) - digamma_t(tc) + torch.log(tc) + 1.0
                 - torch.log(tc + mu) - (Y + tc) / (mu + tc)).sum(1)
        bad = (info == 0) | ~torch.isfinite(info)
        return torch.where(bad, torch.zeros_like(score),
                           score / torch.where(bad, torch.ones_like(info), info)), bad

    def mm_step(t):
        tc = t[:, None]
        den = ((Y - mu) ** 2 / (mu + tc) ** 2).sum(1)
        num = ((Y - mu) ** 2 / (mu + mu ** 2 / tc)).sum(1) - dfr
        bad = (den == 0) | ~torch.isfinite(den)
        return torch.where(bad, torch.zeros_like(num),
                           -num / torch.where(bad, torch.ones_like(den), den)), bad

    t_mle, warn_mle = newton(mle_step, t0)
    est = t_mle.clone()
    if warn_mle.any():
        t_mm, warn_mm = newton(mm_step, t0)
        est = torch.where(warn_mle & ~warn_mm, t_mm, est)      # MM where MLE warned
        est = torch.where(warn_mle & warn_mm, t0, est)         # else pilot
    return est.clamp(THETA_LO, THETA_HI)


def precompute_gene_batch_torch(Y, X, device="cpu", dtype=None, gene_chunk=256):
    """Batched equivalent of precompute_gene_batch, on `device` in `dtype` (default
    float64). Y (G,n) counts, X (n,d) covariates. Returns torch tensors
    A,W (G,n); D (G,d,n); MU (G,n) on `device`. Genes are processed in chunks of
    `gene_chunk` to bound memory; results stay resident on the device."""
    import torch
    if dtype is None:
        dtype = torch.float64
    Xt = torch.as_tensor(X, dtype=dtype, device=device)
    n, d = Xt.shape
    G = Y.shape[0]
    dfr = n - d
    A = torch.empty(G, n, dtype=dtype, device=device)
    W = torch.empty(G, n, dtype=dtype, device=device)
    D = torch.empty(G, d, n, dtype=dtype, device=device)
    MU = torch.empty(G, n, dtype=dtype, device=device)
    for s in range(0, G, gene_chunk):
        e = min(s + gene_chunk, G)
        Yc = torch.as_tensor(Y[s:e], dtype=dtype, device=device)
        mu = _poisson_irls_batched(Yc, Xt)
        theta = _theta_batched(Yc, mu, dfr)
        denom = 1.0 + mu / theta[:, None]
        w = mu / denom
        a = (Yc - mu) / denom
        wZ = w.unsqueeze(-1) * Xt.unsqueeze(0)                 # (g,n,d)
        ZtwZ = torch.einsum('gn,ni,nj->gij', w, Xt, Xt)        # (g,d,d)
        vals, U = torch.linalg.eigh(ZtwZ)
        Dc = torch.bmm((U / torch.sqrt(vals).unsqueeze(1)).transpose(-1, -2),
                       wZ.transpose(-1, -2))                    # (g,d,n)
        A[s:e], W[s:e], D[s:e], MU[s:e] = a, w, Dc, mu
    return A, W, D, MU
