"""
Show that the null score statistic equals a batched matrix-multiply formulation.

The per-resample statistic (per resample k with treated-cell index set S_k) is
    lower_right = sum_i ( sum_{c in S_k} D[i,c] )^2      # i over the d rows of D
    top         = sum_{c in S_k} a[c]
    lower_left  = sum_{c in S_k} w[c]
    z_k         = top / sqrt(lower_left - lower_right)
Writing the B resample sets as a binary membership matrix M (B x n_cells), the whole
vector of null statistics is  z = (M·a) / sqrt(M·w - colSums((D·Mᵀ)²)).

This script checks the two agree to floating-point tolerance — confirming the hot
loop is batched linear algebra, which is what makes it efficient on GPU.
"""
import numpy as np

rng = np.random.default_rng(0)

def naive_null_zscores(a, w, D, resample_sets):
    """Straightforward un-batched reference implementation."""
    out = np.empty(len(resample_sets))
    for k, S in enumerate(resample_sets):
        inner = D[:, S].sum(axis=1)          # length n_cov: sum_{c in S} D[i,c]
        lower_right = float((inner * inner).sum())
        top = float(a[S].sum())
        lower_left = float(w[S].sum())
        out[k] = top / np.sqrt(lower_left - lower_right)
    return out

def matmul_null_zscores(a, w, D, M):
    """Batched reformulation. M is (B x n_cells) binary membership matrix."""
    top = M @ a                               # (B,)
    lower_left = M @ w                         # (B,)
    DM = D @ M.T                               # (n_cov x B)
    lower_right = (DM * DM).sum(axis=0)        # (B,)
    return top / np.sqrt(lower_left - lower_right)

def build_M(resample_sets, n_cells):
    M = np.zeros((len(resample_sets), n_cells))
    for k, S in enumerate(resample_sets):
        M[k, S] = 1.0
    return M

# ---- realistic-ish CRT setup: variable treated-cell count per resample -------
for trial, (n_cells, n_cov, B) in enumerate([(2000, 8, 499), (50000, 12, 5498), (200000, 15, 999)]):
    a = rng.normal(size=n_cells)
    w = rng.uniform(2.0, 5.0, size=n_cells)      # kept > lower_right so variance > 0
    D = rng.normal(size=(n_cov, n_cells)) * 0.01  # small, like eigen-scaled distillation D
    # CRT-style resamples: each cell independently "treated" w.p. p_c
    p = rng.uniform(0.002, 0.02, size=n_cells)
    resample_sets = [np.nonzero(rng.random(n_cells) < p)[0] for _ in range(B)]

    z_naive = naive_null_zscores(a, w, D, resample_sets)
    M = build_M(resample_sets, n_cells)
    z_matmul = matmul_null_zscores(a, w, D, M)

    max_abs = np.max(np.abs(z_naive - z_matmul))
    max_rel = np.max(np.abs(z_naive - z_matmul) / (np.abs(z_naive) + 1e-30))
    avg_trt = np.mean([len(s) for s in resample_sets])
    print(f"trial {trial}: n_cells={n_cells:>6} n_cov={n_cov:>2} B={B:>4} "
          f"avg_n_trt={avg_trt:7.1f} | max|abs diff|={max_abs:.2e} max|rel diff|={max_rel:.2e} "
          f"| {'IDENTICAL' if max_abs < 1e-9 else 'MISMATCH'}")

print("\nConclusion: SCEPTRE's exact null statistic == (M@a)/sqrt(M@w - colSums((D@M^T)^2)).")
print("The genome-scale hot loop is batched (sparse) matmul. Device-independent.")
