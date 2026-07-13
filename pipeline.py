"""
End-to-end screen runner: counts -> precompute -> resample -> kernel -> results.

Ties precompute.py and gpu_sceptre.py into a single `run_screen()` entry point,
runnable on CPU or CUDA. Self-contained (NumPy/SciPy/PyTorch only).
"""
import numpy as np
import torch
import gpu_sceptre as gs
from precompute import precompute_gene_batch, precompute_gene_batch_torch


def run_screen(Y, X, grna_to_cells, device="cpu",
               B1=gs.B1_DEFAULT, B2=gs.B2_DEFAULT, B3=0,
               side=0, gene_chunk=None, seed=0, dtype=torch.float64,
               precompute_chunk=256):
    """Test every gene against every gRNA in `grna_to_cells`.

    Y (G,n) int counts, X (n,d) covariates (incl. intercept),
    grna_to_cells: {grna_id: array of 0-indexed treated cells}.
    Returns (results dict-of-arrays, precompute_seconds, test_seconds).

    Precompute uses the batched PyTorch path on CUDA (all genes fit at once,
    results stay on-device) and the per-gene NumPy reference on CPU, which is
    faster there. Both produce identical a,w,D up to floating-point tolerance.
    """
    import time
    G, n = Y.shape
    # --- precompute a,w,D per gene ONCE ---------------------------------------
    t0 = time.perf_counter()
    if device == "cuda":
        At, Wt, Dt, MUt = precompute_gene_batch_torch(
            Y, X, device=device, dtype=dtype, gene_chunk=precompute_chunk)
        Yt = torch.as_tensor(Y, dtype=dtype, device=device)
    else:
        A, W, D, MU = precompute_gene_batch(Y, X)
        t = lambda x: torch.as_tensor(x, dtype=dtype, device=device)
        At, Wt, Dt, MUt, Yt = t(A), t(W), t(D), t(MU), t(Y)
    if device in ("mps", "cuda"):
        getattr(torch, device).synchronize()
    t_pre = time.perf_counter() - t0

    grna_ids, gene_idx, p, z, fc, stage = [], [], [], [], [], []
    t0 = time.perf_counter()
    for gi, (grna, cells) in enumerate(grna_to_cells.items()):
        cells = np.asarray(cells, dtype=np.int64)
        pool = gs.permutation_resamples(n, len(cells), B1 + B2 + B3, seed=seed + gi)
        res = gs.run_association_test(
            At, Wt, Dt, Yt, MUt, torch.as_tensor(cells), pool,
            B1=B1, B2=B2, B3=B3, side=side, gene_chunk=gene_chunk)
        grna_ids += [grna] * G
        gene_idx += list(range(G))
        p.append(res["p"]); z.append(res["z_orig"]); fc.append(res["fold_change"]); stage.append(res["stage"])
    if device in ("mps", "cuda"):
        getattr(torch, device).synchronize()
    t_test = time.perf_counter() - t0

    return ({"grna": np.array(grna_ids), "gene": np.array(gene_idx),
             "p": np.concatenate(p), "z_orig": np.concatenate(z),
             "fold_change": np.concatenate(fc), "stage": np.concatenate(stage)},
            t_pre, t_test)


def pick_device():
    """Accuracy-first device selection: f64-capable devices only (CUDA, else CPU).
    MPS is intentionally excluded from auto-selection because it has no float64;
    the validated pipeline is f64. Request MPS explicitly (experimental, f32) if
    you want to benchmark the Apple GPU, but it is not part of the accuracy claim."""
    return "cuda" if torch.cuda.is_available() else "cpu"
