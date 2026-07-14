"""
Compare gpu-sceptre against the reference SCEPTRE R package.

Run benchmark_vs_sceptre.R first (it writes sceptre_timing.txt with SCEPTRE's
wall-clock on its own example dataset), then run this script. It runs gpu-sceptre
on a matched workload — the same gene x cell dimensions and the same number of
pairs — and prints the wall-clock comparison.

    Rscript benchmark_vs_sceptre.R      # SCEPTRE side -> sceptre_timing.txt
    python3 benchmark_vs_sceptre.py     # gpu-sceptre side + comparison

Honesty notes:
  * Both timings include per-gene precompute.
  * SCEPTRE runs on its real example data; gpu-sceptre runs a matched-size,
    matched-count simulated workload. Timing depends on dimensions, not values,
    so this is a fair scale comparison; it is NOT an identical-pairs benchmark
    (SCEPTRE's negative-control pairs vs simulated pairs differ).
  * Both are timed on whatever hardware you run them on; report it.
"""
import torch
import pipeline as pipe
from demo import simulate

with open("sceptre_timing.txt") as fh:
    v = fh.read().split()
t_1core = float(v[0])
t_2core = float("nan") if v[1] == "NA" else float(v[1])
n_pairs, genes, cells = int(v[2]), int(v[3]), int(v[4])

device = "cuda" if torch.cuda.is_available() else "cpu"
n_grnas = max(2, round(n_pairs / genes))          # match SCEPTRE's pair count
Y, X, grna_to_cells, _ = simulate(n=cells, G=genes, K_nt=n_grnas, K_pos=0)
res, t_pre, t_test = pipe.run_screen(Y, X, grna_to_cells, device=device,
                                     dtype=torch.float64, B1=499, B2=1999, B3=0,
                                     gene_chunk=64)
ours = t_pre + t_test
our_pairs = genes * len(grna_to_cells)

print(f"\nDataset: {genes} genes x {cells} cells | ~{n_pairs} pairs | both include precompute")
print(f"  gpu-sceptre ({device:4s})   : {ours:7.1f} s  ({our_pairs} pairs)")
print(f"  SCEPTRE   1 CPU core   : {t_1core:7.1f} s   -> {t_1core/ours:5.1f}x slower than gpu-sceptre")
if t_2core == t_2core:  # not NaN
    eff = 100.0 * (t_1core / t_2core) / 2.0
    print(f"  SCEPTRE   2 CPU cores  : {t_2core:7.1f} s   -> {t_2core/ours:5.1f}x slower "
          f"({t_1core/t_2core:.2f}x from 2 cores = {eff:.0f}% efficient)")
    print(f"\n  At SCEPTRE's per-core rate, matching this run takes ~{t_1core/ours:.0f} cores if "
          f"scaling were perfect; the measured {eff:.0f}% efficiency implies more.")
