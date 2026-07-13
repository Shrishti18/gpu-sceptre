"""
Batched GLM score-test kernel for single-cell CRISPR screens.

Implements the conditional-resampling association test of Barry et al. (the SCEPTRE
method): for each gRNA-gene pair, a negative-binomial GLM score statistic is compared
against B resampled null treatment vectors. This module evaluates that statistic and
the accompanying three-stage screen for a whole batch of genes at once (they share a
gRNA's resample set), which maps the per-pair hot loop onto batched linear algebra
that runs efficiently on CPU or CUDA.

Computation is float64 throughout, matching a double-precision reference to machine
epsilon. Upstream steps (gRNA-to-cell assignment, QC, the per-gene NB GLM that yields
the score-test pieces a/w/D) are provided by `precompute.py`.

Reference: Barry, Wang, Roeder, Katsevich (2021), "SCEPTRE improves calibration and
sensitivity in single-cell CRISPR screen analysis", Genome Biology 22:344.

Kernel inputs:
  A   (G, n)     per-gene score-test residual  a = W·M(Y-μ̂)
  W   (G, n)     per-gene weights              w = μ̂/(1+μ̂/θ)
  D   (G, d, n)  per-gene variance factor,     DᵀD = WZ(ZᵀWZ)⁻¹(WZ)ᵀ
  trt_idx (n_trt,)              treated cells for a gRNA (0-indexed)
  resample_pool (B_tot, n_trt)  shared resample sets (uniform without-replacement)
"""
from __future__ import annotations
import numpy as np
import torch
from scipy.stats import skewnorm

# Default resample counts and the stage-1 significance threshold
B1_DEFAULT, B2_DEFAULT, B3_PERM = 499, 4999, 24999
P_THRESH = 0.02
SIDE = {"left": -1, "both": 0, "right": 1}


# ----------------------------------------------------------------------------
# The kernel: batched GLM score statistic (float64).  z = top / sqrt(low_l - low_r)
# ----------------------------------------------------------------------------
def score_zscores(A, W, D, idx, gene_chunk=None, _report=None):
    """Null (or observed) z-scores for a batch of G genes over a set of treated-cell
    index vectors `idx` (B, n_trt). Returns (G, B) float64.

    Batched form of the per-resample null statistic:
        top      = Σ_{c∈S} a[g,c]
        low_left = Σ_{c∈S} w[g,c]
        low_right= Σ_i ( Σ_{c∈S} D[g,i,c] )²
        z        = top / sqrt(low_left - low_right)
    The variance `low_left - low_right` is a residual variance; kept in float64.
    """
    G, B, n_trt, d = A.shape[0], idx.shape[0], idx.shape[1], D.shape[1]
    flat = idx.reshape(-1).to(A.device)                      # (B*n_trt,)
    out = torch.empty((G, B), dtype=A.dtype, device=A.device)
    chunk = gene_chunk or G
    n_nonpos = 0
    for s in range(0, G, chunk):
        e = min(s + chunk, G)
        g = e - s
        top = A[s:e].index_select(1, flat).reshape(g, B, n_trt).sum(2)      # (g,B)
        low_l = W[s:e].index_select(1, flat).reshape(g, B, n_trt).sum(2)    # (g,B)
        Dg = D[s:e].index_select(2, flat).reshape(g, d, B, n_trt).sum(3)    # (g,d,B)
        low_r = (Dg * Dg).sum(1)                                            # (g,B)
        var = low_l - low_r                                                 # residual variance
        n_nonpos += int((var <= 0).sum())
        out[s:e] = top / torch.sqrt(var)
    if _report is not None:
        _report["nonpos_var"] = _report.get("nonpos_var", 0) + n_nonpos
    return out


def observed_zscore(A, W, D, trt_idx, _report=None):
    """z_orig for all G genes on the real treatment set. Returns (G,) float64."""
    return score_zscores(A, W, D, trt_idx.view(1, -1), _report=_report).squeeze(1)


# ----------------------------------------------------------------------------
# Empirical p-value
# ----------------------------------------------------------------------------
def empirical_p(z_null, z_orig, side):
    """z_null (G,B), z_orig (G,) -> p (G,).  p_left=(1+#{orig>=null})/(1+B)."""
    B = z_null.shape[1]
    left = (z_orig[:, None] >= z_null).sum(1).to(z_null.dtype)
    right = (z_orig[:, None] <= z_null).sum(1).to(z_null.dtype)
    p_left = (1.0 + left) / (1.0 + B)
    p_right = (1.0 + right) / (1.0 + B)
    if side == -1:
        return p_left
    if side == 1:
        return p_right
    return 2.0 * torch.minimum(p_left, p_right)


# ----------------------------------------------------------------------------
# Skew-normal fit, goodness-of-fit, and tail p-value
# Operates per-gene on the stage-2 (B2) null slice, in numpy/scipy.
# ----------------------------------------------------------------------------
def fit_skew_normal(y):
    """Method-of-moments skew-normal fit. Returns (xi, omega, alpha, m_y, sd_y)."""
    n = y.size
    MAX_GAMMA_1 = 0.995
    m_y = y.mean()
    sd_y = np.sqrt((y * y).mean() - m_y * m_y)
    gamma1 = (((y - m_y) ** 3).sum()) / (n * sd_y ** 3)
    if gamma1 > MAX_GAMMA_1:
        gamma1 = 0.9 * MAX_GAMMA_1
    b = np.sqrt(2.0 / np.pi)
    r = np.copysign(1.0, gamma1) * (2 * abs(gamma1) / (4 - np.pi)) ** (1.0 / 3.0)
    delta = r / (b * np.sqrt(1 + r * r))
    alpha = delta / np.sqrt(1 - delta * delta)
    mu_z = b * delta
    sd_z = np.sqrt(1 - mu_z * mu_z)
    omega = sd_y / sd_z
    xi = m_y - omega * mu_z
    return xi, omega, alpha, m_y, sd_y


def _check_outliers(y_sorted, m_y, sd_y):
    B = y_sorted.size
    R_max = y_sorted[-1] / (m_y + sd_y * np.sqrt(2 * np.log(B)))
    R_min = y_sorted[0] / (m_y - sd_y * np.sqrt(2 * np.log(B)))
    return (R_max <= 1.5) and (R_min <= 1.5)


def _check_sn_tail(y_sorted, xi, omega, alpha):
    n = y_sorted.size
    for i in range(180, 199):
        p = i / 200.0
        idx = min(int(np.ceil(n * p)), n - 1)
        quantile = y_sorted[idx]
        sn_tail_prob = skewnorm.sf(quantile, alpha, loc=xi, scale=omega)
        if sn_tail_prob <= 0:
            return False
        if (1.0 - p) / sn_tail_prob > 2.0:
            return False
    return True


def fit_and_evaluate_skew_normal(z_orig, null, side):
    """Returns p (skew-normal tail p-value) or -1.0 if the fit is rejected."""
    xi, omega, alpha, m_y, sd_y = fit_skew_normal(null)
    if not all(np.isfinite(v) for v in (xi, omega, alpha, m_y, sd_y)):
        return -1.0
    s = np.sort(null)
    median = s[(s.size - 1) // 2]
    check_right = z_orig >= median
    if not _check_outliers(s, m_y, sd_y):
        return -1.0
    if check_right:
        fit_ok = _check_sn_tail(s, xi, omega, alpha)
    else:  # mirror for the left tail
        fit_ok = _check_sn_tail(-s[::-1], -xi, omega, -alpha)
    if not fit_ok:
        return -1.0
    if side == 0:
        p = 2.0 * (skewnorm.sf(z_orig, alpha, loc=xi, scale=omega) if check_right
                   else skewnorm.cdf(z_orig, alpha, loc=xi, scale=omega))
    elif side == 1:
        p = skewnorm.sf(z_orig, alpha, loc=xi, scale=omega)
    else:
        p = skewnorm.cdf(z_orig, alpha, loc=xi, scale=omega)
    return max(float(p), 1.0e-250)


# ----------------------------------------------------------------------------
# Fold change
# ----------------------------------------------------------------------------
def estimate_fold_change(Y, MU, trt_idx):
    """Returns (fold_change (G,), se_fold_change (G,)) float64."""
    trt_idx = trt_idx.to(Y.device)
    yt = Y[:, trt_idx]
    mut = MU[:, trt_idx]
    sum_mu = mut.sum(1)
    fc = yt.sum(1) / sum_mu
    top = ((yt - fc[:, None] * mut) ** 2).sum(1)
    se = torch.sqrt(top / (sum_mu * sum_mu))
    return fc, se


# ----------------------------------------------------------------------------
# Sampler: uniform without-replacement resample sets (the permutation null).
# Only the sampling *distribution* affects the test, not the RNG stream, so these
# are drawn on CPU with NumPy's generator and transferred to the compute device.
# ----------------------------------------------------------------------------
def permutation_resamples(n_cells, n_trt, n_resamples, seed=0):
    """(n_resamples, n_trt) int64 uniform WOR draws of size n_trt from [0, n_cells)."""
    rng = np.random.default_rng(seed)
    out = np.empty((n_resamples, n_trt), dtype=np.int64)
    for k in range(n_resamples):
        out[k] = rng.choice(n_cells, size=n_trt, replace=False)
    return torch.from_numpy(out)


# ----------------------------------------------------------------------------
# Orchestration: the three-stage screen, batched across genes for one gRNA.
# ----------------------------------------------------------------------------
def run_association_test(A, W, D, Y, MU, trt_idx, resample_pool,
                         B1=B1_DEFAULT, B2=B2_DEFAULT, B3=B3_PERM,
                         side=0, fit_parametric_curve=True, gene_chunk=None):
    """Test G genes against one gRNA. Returns dict of (G,) arrays:
    {p, z_orig, fold_change, se_fold_change, stage}.

    resample_pool: (>=B1+B2+B3, n_trt) shared treated-cell index sets.
    """
    G = A.shape[0]
    report = {}
    trt_idx = trt_idx.to(A.device)
    resample_pool = resample_pool.to(A.device)
    z_orig = observed_zscore(A, W, D, trt_idx, _report=report)          # (G,)
    fc, se = estimate_fold_change(Y, MU, trt_idx)

    # Stage 1: empirical p on [0, B1) for every gene.
    null1 = score_zscores(A, W, D, resample_pool[:B1], gene_chunk, report)   # (G,B1)
    p = empirical_p(null1, z_orig, side)                                     # (G,)
    stage = torch.ones(G, dtype=torch.int64)

    surv = (p <= P_THRESH).nonzero(as_tuple=True)[0]                         # survivors
    if surv.numel() > 0:
        zo = z_orig[surv].cpu().numpy()
        sub = (A[surv], W[surv], D[surv])
        # Stage 2: skew-normal on the disjoint [B1, B1+B2) slice, per survivor gene.
        sn_used = torch.zeros(surv.numel(), dtype=torch.bool)
        if fit_parametric_curve and B2 > 0:
            null2 = score_zscores(*sub, resample_pool[B1:B1 + B2], gene_chunk, report).cpu().numpy()
            for i in range(surv.numel()):
                p_sn = fit_and_evaluate_skew_normal(zo[i], null2[i], side)
                if p_sn > -0.5:
                    p[surv[i]] = p_sn
                    stage[surv[i]] = 2
                    sn_used[i] = True
        # Stage 3: empirical on the disjoint [B1+B2, B1+B2+B3) slice for SN failures
        # (or, when B3==0 as in CRT, empirical on the stage-2 slice).
        fell = (~sn_used).nonzero(as_tuple=True)[0]
        if fell.numel() > 0:
            fi = surv[fell]
            subf = (A[fi], W[fi], D[fi])
            if B3 > 0:
                null3 = score_zscores(*subf, resample_pool[B1 + B2:B1 + B2 + B3], gene_chunk, report)
            else:
                null3 = score_zscores(*subf, resample_pool[B1:B1 + B2], gene_chunk, report)
            p[fi] = empirical_p(null3, z_orig[fi], side)
            stage[fi] = 3

    return {
        "p": p.cpu().numpy(),
        "z_orig": z_orig.cpu().numpy(),
        "fold_change": fc.cpu().numpy(),
        "se_fold_change": se.cpu().numpy(),
        "stage": stage.numpy(),
        "_nonpos_var": report.get("nonpos_var", 0),
    }


def to_tensors(A, W, D, Y, MU, device="cpu", dtype=torch.float64):
    """Move the per-gene precomputation arrays onto the compute device.
    The pipeline runs in float64, which the Apple MPS backend does not support;
    use "cpu" or "cuda"."""
    if device == "mps" and dtype == torch.float64:
        raise ValueError("MPS does not support float64; use device 'cpu' or 'cuda'.")
    t = lambda x: torch.as_tensor(x, dtype=dtype, device=device)
    return t(A), t(W), t(D), t(Y), t(MU)
