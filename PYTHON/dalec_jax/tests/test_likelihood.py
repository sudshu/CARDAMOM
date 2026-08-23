"""L6: 31 likelihood terms + total P vs golden mlf output, 120 fixtures.

EDC/likelihood logic is isolated from trajectory chaos by feeding the C
oracle's own pools/fluxes into the JAX mlf2 composition. Assertions:
per-term rel <= 1e-12 (or abs <= 1e-12), sentinels (-inf / zeroed gated
rows) exact.
"""
import numpy as np
import pytest

import dalec_jax  # noqa: F401
from dalec_jax import edcs, oracle_io
from dalec_jax.indices import NOEDCS
from dalec_jax.likelihood import NOLIKELIHOODS, data_prep, mlf2


@pytest.fixture(scope="module")
def results():
    import jax

    if not (oracle_io.GOLDEN / "trajectories/mlf.bin").exists():
        pytest.skip("goldens missing -- run tools/gen_fixtures.py")
    g = oracle_io.trajectory_golden()
    cbf = data_prep.load_cbf(g["meta"]["cbf"])
    edc_cfg = {"n_timesteps": cbf.n_timesteps,
               "dint": edcs.compute_dint(cbf.time),
               "edc_eqf": cbf.edc_eqf,
               "skt_ref_mean": cbf.skt_ref_mean}

    def one(pars, pools, fluxes):
        return mlf2(cbf, edc_cfg, pars, pools, fluxes)

    rec, ML, Pv = jax.vmap(one)(g["params"], g["pools"], g["fluxes"])
    gold = g["mlf"]
    return (np.asarray(rec), np.asarray(ML), np.asarray(Pv),
            gold[:, :NOEDCS], gold[:, NOEDCS:NOEDCS + NOLIKELIHOODS],
            gold[:, -1])


def _close(c, j, rtol=1e-12, atol=1e-12):
    if np.isnan(c) or np.isnan(j):
        return np.isnan(c) and np.isnan(j)
    if np.isinf(c) or np.isinf(j):
        return c == j
    return abs(j - c) <= max(rtol * abs(c), atol)


def test_likelihood_terms(results):
    _, ML, _, _, gML, _ = results
    bad = [(i, k, gML[i, k], ML[i, k])
           for i in range(ML.shape[0]) for k in range(NOLIKELIHOODS)
           if not _close(gML[i, k], ML[i, k])]
    assert not bad, f"{len(bad)} term mismatches, first 10: {bad[:10]}"


def test_total_P(results):
    _, _, Pv, _, _, gP = results
    bad = [(i, gP[i], Pv[i]) for i in range(len(Pv))
           if not _close(gP[i], Pv[i], rtol=1e-12, atol=1e-9)]
    assert not bad, f"{len(bad)} P mismatches, first 10: {bad[:10]}"


def test_edc_records_via_mlf2(results):
    rec, _, _, gE, _, _ = results
    bad = [(i, k, gE[i, k], rec[i, k])
           for i in range(rec.shape[0]) for k in range(NOEDCS)
           if not _close(gE[i, k], rec[i, k], rtol=1e-10)]
    assert not bad, f"{len(bad)} EDC mismatches, first 10: {bad[:10]}"


def test_gated_rows_zeroed(results):
    """Prerun-gated samples must leave ML all-zero exactly (C skips the
    likelihood; oracle zeroes buffers per sample)."""
    _, ML, _, gE, gML, gP = results
    gated = ~np.isfinite(gP)
    assert gated.any()
    np.testing.assert_array_equal(ML[gated], 0.0)
    np.testing.assert_array_equal(gML[gated], 0.0)
