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
(or CPU) in **float64**, matching a double-precision reference to machine precision.

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

Measured on a single NVIDIA T4 (free tier), simulated screen, float64:

| phase | CPU | GPU (T4) | speedup |
|---|---|---|---|
| precompute — 2000 genes × 50k cells | ~595 s | ~20 s | ~30× |
| resampling kernel — throughput | ~245 pairs/s | ~7,400 pairs/s | ~30× |

The same code runs on either device — only the `device` string changes — and produces
identical float64 results. On CUDA the batched precompute keeps `a, w, D` resident on the
GPU, so nothing round-trips to the host between phases. End to end at that scale the screen
goes from ~600 s (CPU-bound precompute) to ~23 s — roughly 26×. Numbers are workload-dependent;
reproduce them with `demo.py`.

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
| `test_gpu_sceptre.py` | correctness tests against an independent reference |
| `reformulation_proof.py` | matrix-multiply-equivalence check |

## Devices and precision

- **CPU** and **CUDA** run in float64 — the validated path.
- The Apple **MPS** backend has no float64; MPS is supported only in float32 as an
  experimental speed option and is not used for reference-accuracy results.

## Citation

This implements the statistical method of:

> Barry T., Wang X., Roeder K., Katsevich E. (2021).
> *SCEPTRE improves calibration and sensitivity in single-cell CRISPR screen analysis.*
> Genome Biology 22, 344.

Reference implementation: <https://github.com/Katsevich-Lab/sceptre> (GPL-3.0).

## License

GPL-3.0 — see [LICENSE](LICENSE).
