"""L4: full 240-step trajectories vs the C oracle, all 120 fixtures.

Element criterion: |jax - c| <= 1e-10 * max(|c|, RMS_var)  OR  |jax - c| <=
1e-12 absolute (the absolute escape covers catastrophic-cancellation
diagnostics like hydraulic_mortality_factor ~ 1e-15 whose entire dynamic
range is rounding noise).

Fixture gate:
  CLEAN            — every element passes, break step (if any) matches C
                     exactly, post-break zero tail exact.
  CHAOS-CERTIFIED  — the fixture fails pointwise only if the C model ITSELF
                     fails pointwise against 1-ULP-perturbed parameter
                     vectors (K=8 all-parameter dithers, run through the C
                     oracle at golden-generation time), and the JAX
                     divergence onset is no earlier than the C self-
                     divergence onset minus MARGIN steps. Justification and
                     measurements: TOLERANCES.md "Trajectory chaos" section.

Aggregate guards: every fixture whose C self-onset is -1 (C not 1-ULP-
sensitive) must be CLEAN (enforced per fixture), and the number of clean
fixtures must be at least the number of non-sensitive fixtures — chaos
certificates can never exceed the C model's own measured ULP sensitivity.
"""
import json

import numpy as np
import pytest

import dalec_jax  # noqa: F401
from dalec_jax import oracle_io

REL_TOL = 1e-10
ABS_TOL = 1e-12
MARGIN = 10


def _element_fail(c, j):
    fin = np.isfinite(c)
    with np.errstate(all="ignore"):
        rms = np.sqrt(np.nanmean(np.where(fin, c, np.nan) ** 2, axis=0))
        rms = np.nan_to_num(rms, nan=1.0)
        mixed = np.abs(j - c) / np.maximum(np.abs(np.where(fin, c, 0.0)),
                                           np.maximum(rms, 1e-300))
    ok = (mixed <= REL_TOL) | (np.abs(j - c) <= ABS_TOL)
    return np.where(fin & np.isfinite(j), ~ok,
                    ~((~np.isfinite(c)) & (~np.isfinite(j))))


def _tdiv(cp, jp, cf, jf):
    fp = _element_fail(cp, jp).any(axis=1)
    ff = _element_fail(cf, jf).any(axis=1)
    rows = fp | np.concatenate([ff, [False]])
    return int(np.argmax(rows)) if rows.any() else -1


def _break_step(pools):
    zero = np.all(pools == 0, axis=-1)
    return int(np.argmax(zero)) if zero.any() else -1


@pytest.fixture(scope="module")
def jax_runs():
    import netCDF4
    import jax
    from dalec_jax.model import prederive_vegk, run_dalec_1100

    if not (oracle_io.GOLDEN / "trajectories/pools.bin").exists():
        pytest.skip("goldens missing -- run tools/gen_fixtures.py")
    g = oracle_io.trajectory_golden()
    with netCDF4.Dataset(g["meta"]["cbf"]) as ds:
        met = {k: np.array(ds[k][:], dtype=float) for k in
               ("SSRD", "T2M_MIN", "T2M_MAX", "CO2", "DOY", "TOTAL_PREC",
                "VPD", "BURNED_AREA", "SNOWFALL", "SKT", "STRD",
                "DISTURBANCE_FLUX", "YIELD")}
        tix = np.array(ds["time"][:], dtype=float)
        LAT = float(ds["LAT"][:])
    deltat = float(tix[1] - tix[0])
    VegK = prederive_vegk(met["DOY"], LAT)
    run = jax.jit(lambda p: run_dalec_1100(p, met, LAT, deltat, VegK))
    P, X = jax.vmap(run)(g["params"])
    cert = json.loads(
        (oracle_io.GOLDEN / "trajectories/chaos_cert.json").read_text())
    return g, np.asarray(P), np.asarray(X), cert["min_self_divergence_step"]


def test_trajectories(jax_runs):
    g, P, X, self_onsets = jax_runs
    n = P.shape[0]
    clean = chaotic = 0
    failures = []
    for i in range(n):
        cp, cf = g["pools"][i], g["fluxes"][i]
        tj = _tdiv(cp, P[i], cf, X[i])
        kind = g["meta"]["rows"][i]["kind"]
        if tj < 0:
            bc, bj = _break_step(cp), _break_step(P[i])
            if bc != bj:
                failures.append((i, kind, f"clean but break step {bj} != C {bc}"))
            else:
                clean += 1
            continue
        tc = self_onsets[i]
        if tc >= 0 and tj >= tc - MARGIN:
            chaotic += 1
        else:
            failures.append(
                (i, kind, f"diverges at t={tj}, C self-onset={tc} — NOT "
                          f"explainable by 1-ULP sensitivity"))
    assert not failures, "\n".join(map(str, failures))
    n_insensitive = sum(1 for t in self_onsets if t < 0)
    assert clean >= n_insensitive, \
        (f"{clean}/{n} clean but C is 1-ULP-insensitive on {n_insensitive} — "
         f"chaos certificates exceed the C's own sensitivity "
         f"(chaos-certified: {chaotic})")


def test_zero_tail_and_poisoned_step(jax_runs):
    """C break semantics: poisoned step recorded, calloc-zero tail after."""
    g, P, X, _ = jax_runs
    checked = 0
    for i in range(P.shape[0]):
        bc = _break_step(g["pools"][i])
        bj = _break_step(P[i])
        if bj < 0:
            continue
        assert np.all(P[i, bj:] == 0.0), f"fixture {i}: nonzero pool tail"
        assert np.all(X[i, bj - 1:] == 0.0), f"fixture {i}: nonzero flux tail"
        assert not np.isfinite(P[i, bj - 1]).all(), \
            f"fixture {i}: break at {bj} but step {bj-1} is finite"
        if bc == bj:
            checked += 1
    assert checked >= 40, f"only {checked} matching-break fixtures checked"
