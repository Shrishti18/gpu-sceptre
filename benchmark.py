"""
Reproducible scaling benchmark.

Runs the full screen (batched precompute + resampling kernel) at several problem
sizes on the chosen device and reports, per size: precompute time, kernel time,
throughput (gene-gRNA pairs/s), and peak GPU memory. It then reports normalized
per-unit rates (flat rates => roughly linear scaling) and a clearly-labeled
extrapolation to genome scale.

Every number the README quotes for performance is reproducible with this script:

    python3 benchmark.py --device cuda           # GPU sweep (recommended)
    python3 benchmark.py --device cuda --cpu-baseline   # also time one size on CPU
    python3 benchmark.py --device cpu            # CPU sweep (slow)

Honesty notes printed by the script:
  * The CPU path is this implementation's own pure-Python precompute, NOT the
    SCEPTRE R package (compiled C/Fortran, and faster than this CPU path). The
    CPU-vs-GPU numbers isolate the GPU speedup within one codebase.
  * Genome-scale timing is an EXTRAPOLATION from the measured linear rates, not a
    run; the dense float64 count matrix at that size does not fit a typical GPU.
"""
import argparse, statistics, time
import numpy as np
import torch
import pipeline as pipe
from demo import simulate

# (genes, cells, non-targeting gRNAs); +4 targeting gRNAs are added by simulate()
CONFIGS = [
    (500, 20000, 8),
    (1000, 30000, 12),
    (2000, 50000, 12),
    (3000, 50000, 16),
    (1500, 100000, 12),
]
B1, B2 = 499, 1999
GENOME = (20000, 200000, 2_500_000)   # genes, cells, pairs — for the extrapolation only


def run_one(genes, cells, grnas, device, dtype):
    Y, X, grna_to_cells, _ = simulate(n=cells, G=genes, K_nt=grnas)
    res, t_pre, t_ker = pipe.run_screen(Y, X, grna_to_cells, device=device, dtype=dtype,
                                        B1=B1, B2=B2, B3=0, gene_chunk=64)
    pairs = genes * len(grna_to_cells)
    return t_pre, t_ker, pairs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None, help="cuda | cpu (default: cuda if available)")
    ap.add_argument("--cpu-baseline", action="store_true",
                    help="also time the 2000x50k size on CPU (slow) for the speedup number")
    args = ap.parse_args()
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float64
    is_cuda = device == "cuda"
    name = torch.cuda.get_device_name(0) if is_cuda else "CPU"
    print(f"device={device} ({name})  dtype=float64  torch={torch.__version__}")
    if not is_cuda:
        print("note: CPU sweep is slow; the GPU sweep is the intended use.")
    print()

    # warmup so kernel compilation does not land on the first timed size
    Yw, Xw, gw, _ = simulate(n=3000, G=100, K_nt=4)
    pipe.run_screen(Yw, Xw, gw, device=device, dtype=dtype, B1=B1, B2=B2, B3=0, gene_chunk=64)
    if is_cuda:
        torch.cuda.synchronize()

    rows = []
    hdr = f"{'genes':>6}{'cells':>8}{'gRNAs':>7}{'pairs':>9}{'pre_s':>8}{'ker_s':>8}{'pairs/s':>10}"
    hdr += f"{'GB':>6}" if is_cuda else ""
    print(hdr)
    for genes, cells, grnas in CONFIGS:
        if is_cuda:
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        try:
            t_pre, t_ker, pairs = run_one(genes, cells, grnas, device, dtype)
            n_grnas = pairs // genes
            mem = torch.cuda.max_memory_allocated() / 1e9 if is_cuda else 0.0
            rows.append((genes, cells, n_grnas, pairs, t_pre, t_ker, pairs / t_ker, mem))
            line = f"{genes:6d}{cells:8d}{n_grnas:7d}{pairs:9d}{t_pre:8.1f}{t_ker:8.1f}{pairs/t_ker:10.0f}"
            line += f"{mem:6.1f}" if is_cuda else ""
            print(line)
        except RuntimeError as e:
            print(f"{genes:6d}{cells:8d}  -> skipped ({str(e)[:44]})")
        if is_cuda:
            torch.cuda.empty_cache()

    if not rows:
        print("\nno sizes completed."); return

    print("\nnormalized rates (roughly flat => ~linear scaling):")
    pre_rates, ker_rates = [], []
    for g, c, gr, p, tp, tk, pps, mem in rows:
        pre_rates.append(tp / (g * c)); ker_rates.append(pps)
        print(f"  {g:5d} x {c:6d}: precompute {tp/(g*c)*1e9:6.1f} ns/(gene*cell)  |  kernel {pps:8.0f} pairs/s")

    if args.cpu_baseline and is_cuda:
        print("\ntiming 2000 x 50000 on CPU for the speedup reference (slow)...")
        tp_cpu, tk_cpu, pairs_cpu = run_one(2000, 50000, 12, "cpu", dtype)
        gpu_row = next((r for r in rows if r[0] == 2000 and r[1] == 50000), None)
        if gpu_row:
            print(f"  precompute: {tp_cpu:.0f}s CPU  vs {gpu_row[4]:.1f}s GPU  -> {tp_cpu/gpu_row[4]:.0f}x")
            print(f"  kernel    : {pairs_cpu/tk_cpu:.0f} pairs/s CPU  vs {gpu_row[6]:.0f} pairs/s GPU"
                  f"  -> {gpu_row[6]/(pairs_cpu/tk_cpu):.0f}x")
        print("  (CPU here = this repo's pure-Python path, NOT the compiled SCEPTRE R package.)")

    # extrapolation — clearly labeled, with the memory reality stated
    pre_r, ker_r = statistics.median(pre_rates), statistics.median(ker_rates)
    GG, CC, PP = GENOME
    pre_ext, ker_ext = pre_r * GG * CC, PP / ker_r
    dense_gb = GG * CC * 8 / 1e9
    print(f"\n=== EXTRAPOLATION to genome scale ({GG} genes, {CC} cells, {PP:,} pairs) ===")
    print(f"  precompute ~ {pre_ext/60:.1f} min   kernel ~ {ker_ext/60:.1f} min   total ~ {(pre_ext+ker_ext)/60:.1f} min")
    print( "  [EXTRAPOLATED from the linear rates above -- NOT measured; kernel assumes n_trt ~ fixed]")
    print(f"  memory: a dense float64 count matrix at that size is ~{dense_gb:.0f} GB, which does NOT fit")
    print( "          a T4/free Colab; genome scale needs a large-VRAM GPU or sparse support (not built).")
    print( "  context: the SCEPTRE paper reports ~1 day on ~500 CPU cores for a job of this scale.")


if __name__ == "__main__":
    main()
