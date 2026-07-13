"""
End-to-end demonstration on simulated data (self-contained).

Simulates a small single-cell CRISPR screen and reports:
  (1) CALIBRATION — non-targeting (null) p-values are uniform  -> QQ-plot PNG.
  (2) POWER       — planted knockdowns are detected (small p at the target gene).
  (3) SPEED       — per-pair vs batched kernel wall-clock on the selected device.

Usage:
  python3 demo.py                 # auto-detects device (CUDA if present, else CPU)
  python3 demo.py --device cuda   # NVIDIA GPU (float64)
  python3 demo.py --n 50000 --genes 2000 --skip-baseline   # larger timing run
"""
import argparse, time
import numpy as np
import torch
import gpu_sceptre as gs
import pipeline as pipe

def simulate(n=1200, G=400, d=3, K_nt=8, K_pos=4, n_trt=120, effect=0.4, seed=1):
    rng = np.random.default_rng(seed)
    # covariates: intercept, a library-size-like term, a batch-like term
    X = np.column_stack([np.ones(n), rng.normal(0, 1, n), rng.normal(0, 1, n)])
    coefs = np.column_stack([rng.uniform(-1.0, 1.5, G), rng.normal(0, 0.3, G), rng.normal(0, 0.3, G)])
    mu = np.exp(coefs @ X.T)                        # (G,n) NB mean, covariate-driven
    theta = rng.uniform(5, 50, G)
    Y = rng.negative_binomial(theta[:, None], theta[:, None] / (theta[:, None] + mu)).astype(float)

    grna_to_cells, planted = {}, []
    for k in range(K_nt):                            # non-targeting -> null (calibration)
        grna_to_cells[f"nt_{k}"] = rng.choice(n, n_trt, replace=False)
    for k in range(K_pos):                           # targeting -> plant a real knockdown (power)
        cells = rng.choice(n, n_trt, replace=False)
        tg = rng.integers(G)
        m = mu[tg, cells] * effect                   # knock the target gene down in treated cells
        Y[tg, cells] = rng.negative_binomial(theta[tg], theta[tg] / (theta[tg] + m))
        grna_to_cells[f"pos_{k}"] = cells
        planted.append((f"pos_{k}", tg))
    return Y, X, grna_to_cells, planted

def qq_png(null_p, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    p = np.sort(null_p)
    exp = -np.log10((np.arange(1, len(p) + 1) - 0.5) / len(p))
    obs = -np.log10(np.clip(p, 1e-12, 1))
    fig, ax = plt.subplots(figsize=(5, 5))
    lim = max(exp.max(), obs.max()) * 1.05
    ax.plot([0, lim], [0, lim], color="#0d9488", lw=1.5, label="calibrated (y=x)")
    ax.scatter(exp, obs, s=10, color="#334155", alpha=0.7)
    ax.set_xlabel("expected  -log10(p)  [Uniform]"); ax.set_ylabel("observed  -log10(p)  [gpu_sceptre]")
    ax.set_title("Calibration: null (non-targeting) p-values"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default=None, choices=[None, "float32", "float64"])
    ap.add_argument("--n", type=int, default=1200, help="cells")
    ap.add_argument("--genes", type=int, default=400)
    ap.add_argument("--grnas", type=int, default=8, help="# non-targeting gRNAs (calibration)")
    ap.add_argument("--b1", type=int, default=499)
    ap.add_argument("--b2", type=int, default=1999)
    ap.add_argument("--skip-baseline", action="store_true", help="skip per-pair baseline (for big/GPU runs)")
    args = ap.parse_args()
    dev = args.device or pipe.pick_device()
    dtype = torch.float32 if (args.dtype == "float32" or dev == "mps") else torch.float64
    if dev == "mps":
        print("note: MPS runs in float32 (the backend has no float64); use cpu/cuda for the "
              "reference float64 results.\n")
    print(f"device={dev}  dtype={str(dtype).split('.')[-1]}  torch={torch.__version__}  "
          f"| n={args.n} genes={args.genes} grnas={args.grnas + 4} B1={args.b1}\n")

    Y, X, grna_to_cells, planted = simulate(n=args.n, G=args.genes, K_nt=args.grnas)
    G = Y.shape[0]; B1, B2, B3 = args.b1, args.b2, 0

    # ---- correctness/calibration+power on the chosen device ------------------
    res, t_pre, t_test = pipe.run_screen(Y, X, grna_to_cells, device=dev, dtype=dtype,
                                         B1=B1, B2=B2, B3=B3, gene_chunk=64)
    is_nt = np.char.startswith(res["grna"].astype(str), "nt_")
    null_p = res["p"][is_nt]
    # calibration metric: fraction of null p below 0.05 should be ~0.05; and KS-ish mean~0.5
    frac05 = float(np.mean(null_p <= 0.05)); meanp = float(np.mean(null_p))
    qq_png(null_p, "calibration_qq.png")

    # power: p at each planted (gRNA,target)
    pw = []
    for grna, tg in planted:
        m = (res["grna"] == grna) & (res["gene"] == tg)
        pw.append(float(res["p"][m][0]))

    print("== (1) CALIBRATION (non-targeting nulls) ==")
    print(f"   {is_nt.sum()} null pairs | mean p = {meanp:.3f} (want ~0.5) | "
          f"frac(p<=0.05) = {frac05:.3f} (want ~0.05)  ->  {'CALIBRATED' if 0.02<=frac05<=0.09 else 'CHECK'}")
    print(f"   QQ-plot written: calibration_qq.png\n")
    print("== (2) POWER (planted knockdowns, effect x0.4) ==")
    for (grna, tg), pv in zip(planted, pw):
        print(f"   {grna} -> gene {tg}:  p = {pv:.2e}  {'DETECTED' if pv < 0.05 else 'missed'}")
    print()

    # ---- (3) SPEED: per-pair baseline vs batched, on this device ------------
    print("== (3) SPEED (same workload) ==")
    n_pairs = G * len(grna_to_cells)
    if not args.skip_baseline:
        _, _, t_base = pipe.run_screen(Y, X, grna_to_cells, device=dev, dtype=dtype,
                                       B1=B1, B2=B2, B3=B3, gene_chunk=1)     # per-pair (un-batched)
        print(f"   per-pair (gene_chunk=1) : {t_base:7.2f}s  ({n_pairs/t_base:8.0f} pairs/s)")
    print(f"   batched  (gene_chunk=64): {t_test:7.2f}s  ({n_pairs/t_test:8.0f} pairs/s)"
          + (f"  -> {t_base/t_test:.1f}x from batching on {dev}" if not args.skip_baseline else f"  on {dev}"))
    print(f"   precompute (once)       : {t_pre:7.2f}s")
    print(f"\n   Re-run with --device cuda on a GPU to benchmark accelerated throughput.")

if __name__ == "__main__":
    main()
