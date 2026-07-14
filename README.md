# gpu-sceptre

GPU-accelerated conditional-resampling association testing for single-cell CRISPR screens.

Single-cell CRISPR screens (Perturb-seq) ask whether perturbing a genomic element changes
the expression of a gene. Testing this with **calibrated** statistics — controlling false
positives despite sparsity and technical confounding — is what the SCEPTRE method
(Barry et al., 2021) provides: a negative-binomial GLM **score test** compared against
**conditionally-resampled** null treatment vectors. The cost is compute — at genome scale
the test runs over millions of gRNA–gene pairs, each with hundreds to thousands of resamples.

`gpu-sceptre` is an independent implementation in Python/PyTorch. Both halves of the cost —
the per-gene negative-binomial GLM **precompute** and the per-pair resampling **kernel** — are
expressed as **batched tensor operations across genes**, so the whole screen runs on one GPU
(or CPU) in **float64**. It matches an independent double-precision reference to floating-point
tolerance (the kernel to ~1e-15; the precompute pieces to ~1e-10) — see Accuracy.

## Method

For each gene, a negative-binomial GLM is fit (Poisson fit + residual-based dispersion) to
produce the score-test pieces `a, w, D` — either per-gene in NumPy, or batched across all
genes as tensor operations for GPU execution. For each gRNA, `B` without-replacement null
treatment sets are drawn. The null score statistic for the whole gene batch is then

```
z = (M · a) / sqrt(M · w  −  colSums((D · Mᵀ)²))
```

where `M` is the (B × cells) resample-membership matrix. A three-stage screen (empirical
p-value → skew-normal tail approximation → empirical escalation) yields precise small p-values
from a limited number of resamples.

Based on the statistical method of Barry, Wang, Roeder & Katsevich (2021), *Genome Biology*.

## Install

```bash
pip install -r requirements.txt        # numpy, scipy, torch (+ matplotlib for the demo)
```

## Quickstart

```python
import numpy as np
import pipeline

# Y : (genes, cells) integer counts
# X : (cells, covariates) design matrix, including an intercept column
# grna_to_cells : {grna_id: array of 0-indexed treated cells}

results, t_precompute, t_test = pipeline.run_screen(Y, X, grna_to_cells, device="cuda")  # or "cpu"
# results: dict of arrays -> grna, gene, p, z_orig, fold_change, stage
```

## Scope

`gpu-sceptre` is the statistical analysis: count matrix + covariates + guide→cell mapping
→ calibrated p-values, effect sizes, and stages. Producing those inputs from raw sequencer
output (loading a count matrix, computing covariates like library size, assigning guides to
cells) is standard Perturb-seq preprocessing — typically done with scanpy/AnnData and often
already present in a dataset. That preprocessing is out of scope here; it is **not** a
dependency on the SCEPTRE R package. A convenience AnnData loader is not yet included.

## Demo

```bash
python3 demo.py                      # simulate a screen; report calibration, power, speed
python3 demo.py --device cuda --n 50000 --genes 2000 --skip-baseline   # larger GPU timing run
```

The demo prints and verifies three things on simulated data:

- **Calibration** — non-targeting (null) p-values are uniform; writes `calibration_qq.png`.
- **Power** — planted knockdowns are recovered at small p-values.
- **Speed** — per-pair vs batched throughput on the selected device.

On CPU it reports well-calibrated nulls (mean p ≈ 0.50, fraction below 0.05 ≈ 0.05) and a
few-fold speedup from batching alone. On a GPU both phases are accelerated (see Performance).

## Performance

All numbers below are **measured on a single free-tier NVIDIA T4**, float64, on simulated
data, and are reproducible with `python3 benchmark.py --device cuda`.

Scaling on the T4 (batched precompute + resampling kernel):

| genes | cells | pairs | precompute | kernel | pairs/s | peak GPU mem |
|---|---|---|---|---|---|---|
| 500  | 20k  | 6k  | 2.6 s  | 1.2 s | ~5,000 | 1.1 GB |
| 1000 | 30k  | 16k | 7.3 s  | 2.8 s | ~5,800 | 2.4 GB |
| 2000 | 50k  | 32k | 16.6 s | 4.2 s | ~7,500 | 6.5 GB |
| 3000 | 50k  | 60k | 28.8 s | 7.6 s | ~7,900 | 8.9 GB |
| 1500 | 100k | 24k | 19.6 s | 4.3 s | ~5,600 | 10.5 GB |

Per-unit cost is roughly linear (and improves with size). For the same 2000×50k screen, this
implementation's **CPU** path takes ~590 s (precompute) and runs the kernel at ~245 pairs/s —
so the GPU is roughly **30× faster per phase**.

> **What the CPU column is:** the CPU numbers are *this implementation's own pure-Python
> precompute*, **not** the SCEPTRE R package (which is compiled C/Fortran and faster than this
> CPU path). These figures isolate the GPU speedup within one codebase; they are **not** a
> benchmark against SCEPTRE. A like-for-like comparison against the R package has not been run.

**Genome scale is not measured.** Extrapolating the linear rates above to 20,000 genes ×
200,000 cells (~2.5M pairs) gives roughly ~20 min of GPU compute — but this is an
extrapolation, not a run. The dense float64 count matrix at that size is ~32 GB, which does
**not** fit a T4 or free Colab; genome scale would need a large-VRAM GPU or sparse-matrix
support (not yet implemented). For context, the SCEPTRE paper reports ~1 day on ~500 CPU
cores for a job of that scale.

## Accuracy

Computation is float64 end to end.

- `python3 test_gpu_sceptre.py` checks the batched kernel against an independent NumPy
  reference across the full three-stage screen (agreement to ~1e-15, exact stage assignment),
  and checks the batched precompute against the per-gene NumPy reference.
- `python3 reformulation_proof.py` verifies the batched matrix-multiply form equals the
  per-resample statistic to floating-point tolerance.

## Layout

| file | purpose |
|---|---|
| `gpu_sceptre.py` | batched score-test kernel, three-stage screen, p-values, sampler |
| `precompute.py`  | negative-binomial GLM precomputation (`a, w, D`) — per-gene NumPy + batched Torch |
| `pipeline.py`    | `run_screen()` — end-to-end counts → results |
| `demo.py`        | calibration, power, and speed on simulated data |
| `benchmark.py`   | reproducible scaling sweep + genome-scale extrapolation |
| `test_gpu_sceptre.py` | correctness tests against an independent reference |
| `reformulation_proof.py` | matrix-multiply-equivalence check |

## Devices and precision

- **CPU** and **CUDA** run in float64 — the validated path.
- The Apple **MPS** backend has no float64. The code accepts an MPS device in float32
  (and blocks mps+float64), but that path is **untested** — there is no Apple hardware
  in the loop — and is not part of the validated results. Use CPU or CUDA for accuracy.

## Citation

This implements the statistical method of:

> Barry T., Wang X., Roeder K., Katsevich E. (2021).
> *SCEPTRE improves calibration and sensitivity in single-cell CRISPR screen analysis.*
> Genome Biology 22, 344.

Reference implementation: <https://github.com/Katsevich-Lab/sceptre> (GPL-3.0).

## License

GPL-3.0 — see [LICENSE](LICENSE).
