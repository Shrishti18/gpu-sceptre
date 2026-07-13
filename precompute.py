"""
Per-gene precomputation for the score test (pure NumPy/SciPy).

Regresses each gene's counts onto the cell covariates and produces the score-test
pieces (a, w, D) consumed by the kernel. Following Barry et al., the negative-binomial
GLM is fit quickly by (i) a Poisson GLM, (ii) estimating the NB size parameter theta
from the Poisson residuals, and (iii) taking the NB coefficients equal to the Poisson
coefficients. This makes the pipeline self-contained: raw counts -> a,w,D -> kernel
-> p-values.

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
    """Y (G,n) counts, X (n,d) covariates (incl. intercept). Returns A,W,D,MU,MU_hat.
    A,W (G,n); D (G,d,n); MU (G,n) NB fitted mean; also returns raw Y as-is upstream."""
    G, n = Y.shape
    d = X.shape[1]
    A = np.empty((G, n)); W = np.empty((G, n)); D = np.empty((G, d, n)); MU = np.empty((G, n))
    for g in range(G):
        coefs, theta = response_precomputation(Y[g], X)
        a, w, Dg, mu = precompute_pieces(Y[g], X, coefs, theta)
        A[g], W[g], D[g], MU[g] = a, w, Dg, mu
    return A, W, D, MU
