"""L7: full JAX pipeline (model -> EDCs -> likelihood -> P) on ALL 4000
posterior samples vs the C oracle's mlf output.

Three-clause per-sample gate (rationale in TOLERANCES.md "Trajectory chaos"
plus the P-level findings recorded in CHANGELOG 2026-08-23):
  (a) EXACT/TIGHT — finiteness of P matches AND (both -inf, or rel diff
      <= 1e-9);
  (b) BOTH-REJECTED — P disagrees but both values are astronomically
      rejected (P < -1e5 on both sides, i.e. posterior weight exp(P) = 0
      either way; the 0/0-adjacent PEQ_CUE ratio for dead-vegetation
      samples lives here);
  (c) CHAOS-CERTIFIED — under K=8 all-parameter 1-ULP dithers (golden
      posterior/dither_P.bin) the C's own P either moves at least 1/10 as
      far as the JAX-C difference, or any dither destabilizes the C's own
      gate (P -> -inf while base is finite, or vice versa). Measured
      example: sample 2523 (base P=-228.19) has dither dP spanning
      [-inf, +0.19] incl. -2.9/-8/-10 — the JAX dP of -3.67 sits inside
      the C's own 1-ULP sensitivity distribution.

Aggregate guard: clause (a) must cover >= 90% of all samples.
"""
import numpy as np
import pytest

import dalec_jax  # noqa: F401
from dalec_jax import edcs, oracle_io
from dalec_jax.indices import NOEDCS
from dalec_jax.likelihood import NOLIKELIHOODS, data_prep, mlf2

REL = 1e-9
REJECTED = -1e5
BATCH = 500


@pytest.fixture(scope="module")
def l7():
    import jax
    from dalec_jax.model import prederive_vegk, run_dalec_1100

    pdir = oracle_io.GOLDEN / "posterior"
    if not (pdir / "mlf.bin").exists():
        pytest.skip("posterior goldens missing -- run tools/gen_fixtures.py")
    g = oracle_io.trajectory_golden()
    pars = np.fromfile(pdir / "params.bin").reshape(-1, 89)
    n = pars.shape[0]
    gold = np.fromfile(pdir / "mlf.bin").reshape(n, NOEDCS + NOLIKELIHOODS + 1)
    ditherP = np.fromfile(pdir / "dither_P.bin").reshape(n, -1)

    cbf = data_prep.load_cbf(g["meta"]["cbf"])
    edc_cfg = {"n_timesteps": cbf.n_timesteps,
               "dint": edcs.compute_dint(cbf.time),
               "edc_eqf": cbf.edc_eqf,
               "skt_ref_mean": cbf.skt_ref_mean}
    VegK = prederive_vegk(cbf.met["DOY"], cbf.LAT)

    def full(p):
        pools, fluxes = run_dalec_1100(p, cbf.met, cbf.LAT, cbf.deltat, VegK)
        return mlf2(cbf, edc_cfg, p, pools, fluxes)

    fj = jax.jit(jax.vmap(full))
    Ps = []
    for s in range(0, n, BATCH):
        _, _, p = fj(pars[s:s + BATCH])
        Ps.append(np.asarray(p))
    return np.concatenate(Ps), gold[:, -1], ditherP


def test_posterior_e2e(l7):
    P, gP, dP = l7
    n = len(P)
    tight = rejected = certified = 0
    failures = []
    for i in range(n):
        c, j = gP[i], P[i]
        if np.isfinite(c) == np.isfinite(j) and (
                not np.isfinite(c)
                or abs(j - c) <= REL * abs(c)):
            tight += 1
            continue
        if (not np.isfinite(c) or c < REJECTED) and \
           (not np.isfinite(j) or j < REJECTED):
            rejected += 1
            continue
        jd = abs(j - c) if (np.isfinite(j) and np.isfinite(c)) else np.inf
        dd = dP[i][np.isfinite(dP[i])]
        moved = np.max(np.abs(dd - c)) if (np.isfinite(c) and len(dd)) else 0.0
        gate_unstable = (np.isfinite(dP[i]) != np.isfinite(c)).any()
        if gate_unstable or moved >= jd / 10:
            certified += 1
            continue
        failures.append((i, float(c), float(j)))
    assert not failures, \
        f"{len(failures)} samples fail all clauses, first 10: {failures[:10]}"
    assert tight >= 0.90 * n, \
        (f"tight-agreement clause covers only {tight}/{n} "
         f"(rejected: {rejected}, certified: {certified})")


def test_prerun_gating_exact(l7):
    """Prerun-EDC-gated samples (parameter-only checks) must gate identically
    in JAX — no chaos excuse exists for parameter inequalities."""
    P, gP, _ = l7
    from dalec_jax.likelihood import data_prep as dp  # noqa: F401
    g = oracle_io.trajectory_golden()
    pars = np.fromfile(oracle_io.GOLDEN / "posterior/params.bin").reshape(-1, 89)
    import jax.numpy as jnp
    from dalec_jax.indices import E, P as PI
    ineq = {E.litcwdtor: (PI.t_lit, PI.t_cwd), E.cwdsomtor: (PI.t_cwd, PI.t_som),
            E.rootwoodtor: (PI.t_root, PI.t_wood),
            E.mr_rates: (PI.rauto_mr_r, PI.rauto_mr_w),
            E.fol2lig_cf: (PI.cf_foliar, PI.cf_ligneous),
            E.relativepsi50: (PI.psi_50HMF, PI.psi_50)}
    fail_pre = np.zeros(len(pars), dtype=bool)
    for big, small in ineq.values():
        fail_pre |= pars[:, big] < pars[:, small]
    # samples failing a prerun inequality must be -inf in BOTH
    assert not np.isfinite(gP[fail_pre]).any()
    assert not np.isfinite(P[fail_pre]).any()
