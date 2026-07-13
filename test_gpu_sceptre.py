"""
Self-test for the kernel — no external data required.

An independent NumPy implementation of the per-resample null loop and the three-stage
screen serves as ground truth; the batched PyTorch kernel must reproduce it to machine
precision, including the full staged orchestration. Also checks the empirical p-value,
fold change, and that the sampler produces valid uniform without-replacement draws.

Run:  python3 test_gpu_sceptre.py
"""
import numpy as np
import torch
import gpu_sceptre as gs

torch.set_default_dtype(torch.float64)
RNG = np.random.default_rng(7)
DEV = "cuda" if torch.cuda.is_available() else "cpu"   # float64 on CPU/CUDA


# ---- independent NumPy reference implementation (per gene) ------------------
def ref_null(a, w, D, idx):
    """Per-resample null loop (reference).  a,w:(n,) D:(d,n) idx:(B,n_trt)."""
    out = np.empty(idx.shape[0])
    for k, S in enumerate(idx):
        inner = D[:, S].sum(1)
        out[k] = a[S].sum() / np.sqrt(w[S].sum() - (inner * inner).sum())
    return out

def ref_emp_p(null, z, side):
    B = null.size
    pl = (1 + (z >= null).sum()) / (1 + B)
    pr = (1 + (z <= null).sum()) / (1 + B)
    return pl if side == -1 else pr if side == 1 else 2 * min(pl, pr)

def ref_run_gene(a, w, D, y, mu, trt, pool, B1, B2, B3, side):
    """Per-gene three-stage screen (reference)."""
    z = ref_null(a, w, D, trt[None, :])[0]                       # observed
    p = ref_emp_p(ref_null(a, w, D, pool[:B1]), z, side); stage = 1
    if p <= gs.P_THRESH:
        sn_used = False
        null2 = ref_null(a, w, D, pool[B1:B1 + B2])              # disjoint B2 slice
        if B2 > 0:
            psn = gs.fit_and_evaluate_skew_normal(z, null2, side)
            if psn > -0.5:
                p, stage, sn_used = psn, 2, True
        if not sn_used:
            null3 = ref_null(a, w, D, pool[B1 + B2:B1 + B2 + B3]) if B3 > 0 else null2
            p = ref_emp_p(null3, z, side); stage = 3
    return z, p, stage


def make_data(G=14, n=3000, d=8, n_trt=60, B1=99, B2=299, B3=499, signal_genes=5):
    a = RNG.normal(size=(G, n))
    w = RNG.uniform(2.0, 5.0, size=(G, n))
    D = RNG.normal(size=(G, d, n)) * 0.01
    mu = RNG.uniform(0.5, 3.0, size=(G, n))
    y = RNG.poisson(mu).astype(float)
    trt = np.sort(RNG.choice(n, size=n_trt, replace=False))
    # inject signal into some genes so pairs survive stage 1 -> exercise stages 2/3
    for g in range(signal_genes):
        a[g, trt] += RNG.uniform(3.0, 7.0)
    pool = gs.permutation_resamples(n, n_trt, B1 + B2 + B3, seed=11).numpy()
    return a, w, D, y, mu, trt, pool, (B1, B2, B3)


def main():
    print(f"device={DEV}  torch={torch.__version__}\n")
    a, w, D, y, mu, trt, pool, (B1, B2, B3) = make_data()
    G = a.shape[0]
    side = 0  # "both", SCEPTRE default

    # ---- 1. kernel: score_zscores == ref_null, machine-eps -----------------
    A, W, Dt, Y, MU = gs.to_tensors(a, w, D, y, mu, device=DEV)
    pool_t = torch.from_numpy(pool)
    z_mod = gs.score_zscores(A, W, Dt, pool_t[:B1]).cpu().numpy()      # (G,B1)
    z_ref = np.stack([ref_null(a[g], w[g], D[g], pool[:B1]) for g in range(G)])
    kerr = np.max(np.abs(z_mod - z_ref))
    print(f"[1] kernel score_zscores vs reference             : max|diff| = {kerr:.2e}  "
          f"{'PASS' if kerr < 1e-9 else 'FAIL'}")

    # ---- 2. observed z ------------------------------------------------------
    zo_mod = gs.observed_zscore(A, W, Dt, torch.from_numpy(trt)).cpu().numpy()
    zo_ref = np.array([ref_null(a[g], w[g], D[g], trt[None, :])[0] for g in range(G)])
    oerr = np.max(np.abs(zo_mod - zo_ref))
    print(f"[2] observed_zscore                             : max|diff| = {oerr:.2e}  "
          f"{'PASS' if oerr < 1e-9 else 'FAIL'}")

    # ---- 3. empirical p -----------------------------------------------------
    p_mod = gs.empirical_p(torch.from_numpy(z_ref), torch.from_numpy(zo_ref), side).numpy()
    p_ref = np.array([ref_emp_p(z_ref[g], zo_ref[g], side) for g in range(G)])
    perr = np.max(np.abs(p_mod - p_ref))
    print(f"[3] empirical_p                                 : max|diff| = {perr:.2e}  "
          f"{'PASS' if perr < 1e-12 else 'FAIL'}")

    # ---- 4. fold change -----------------------------------------------------
    fc_mod, se_mod = gs.estimate_fold_change(Y, MU, torch.from_numpy(trt))
    sm = mu[:, trt].sum(1); fc_ref = y[:, trt].sum(1) / sm
    se_ref = np.sqrt(((y[:, trt] - fc_ref[:, None] * mu[:, trt]) ** 2).sum(1) / sm ** 2)
    ferr = max(np.max(np.abs(fc_mod.cpu().numpy() - fc_ref)),
               np.max(np.abs(se_mod.cpu().numpy() - se_ref)))
    print(f"[4] fold_change + se                            : max|diff| = {ferr:.2e}  "
          f"{'PASS' if ferr < 1e-9 else 'FAIL'}")

    # ---- 5. full three-stage orchestration ---------------------------------
    out = gs.run_association_test(A, W, Dt, Y, MU, torch.from_numpy(trt), pool_t,
                                  B1=B1, B2=B2, B3=B3, side=side)
    ref = [ref_run_gene(a[g], w[g], D[g], y[g], mu[g], trt, pool, B1, B2, B3, side)
           for g in range(G)]
    z_r = np.array([r[0] for r in ref]); p_r = np.array([r[1] for r in ref])
    st_r = np.array([r[2] for r in ref])
    zerr = np.max(np.abs(out["z_orig"] - z_r))
    perr2 = np.max(np.abs(out["p"] - p_r))
    stage_ok = np.array_equal(out["stage"], st_r)
    print(f"[5] run_association_test vs reference           : "
          f"z {zerr:.1e} | p {perr2:.1e} | stages {'match' if stage_ok else 'DIFFER'}  "
          f"{'PASS' if (zerr < 1e-9 and perr2 < 1e-12 and stage_ok) else 'FAIL'}")
    print(f"    stage histogram: {np.bincount(out['stage'], minlength=4)[1:]}  (stage 1/2/3)")

    # ---- 6. sampler is valid uniform WOR -----------------------------------
    S = gs.permutation_resamples(500, 40, 2000, seed=3).numpy()
    rows_unique = all(len(np.unique(r)) == 40 for r in S)
    in_range = S.min() >= 0 and S.max() < 500
    freq = np.bincount(S.reshape(-1), minlength=500)
    unif = freq.max() / freq.mean() < 1.5 and freq.min() / freq.mean() > 0.5
    samp_ok = rows_unique and in_range and unif
    print(f"[6] permutation sampler (WOR, uniform)          : "
          f"unique-rows={rows_unique} in-range={in_range} uniform={unif}  "
          f"{'PASS' if samp_ok else 'FAIL'}")

    ok = (kerr < 1e-9 and oerr < 1e-9 and perr < 1e-12 and ferr < 1e-9
          and zerr < 1e-9 and perr2 < 1e-12 and stage_ok and samp_ok)
    print("\n" + ("ALL PASS - kernel reproduces the staged score test to machine precision."
                  if ok else "FAILURE — see above."))
    if out["_nonpos_var"]:
        print(f"note: {out['_nonpos_var']} non-positive variances encountered (float64 guard).")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
