"""L2+L5: the 15 EDCs vs golden M_EDCs on all 120 fixtures.

EDC logic is isolated from trajectory chaos by feeding the C ORACLE'S OWN
pools/fluxes into the JAX EDC functions; the recorded M_EDCs must then match
the golden mlf output: -inf/0 booleans and short-circuit-masked slots
EXACTLY, finite penalty values to 1e-10 relative (means/sums may differ in
summation-order ULPs), NaN slots by class.
"""
import numpy as np
import pytest

import dalec_jax  # noqa: F401
from dalec_jax import edcs, oracle_io
from dalec_jax.indices import NOEDCS


@pytest.fixture(scope="module")
def edc_results():
    import netCDF4
    import jax
    import jax.numpy as jnp

    if not (oracle_io.GOLDEN / "trajectories/mlf.bin").exists():
        pytest.skip("goldens missing -- run tools/gen_fixtures.py")
    g = oracle_io.trajectory_golden()
    with netCDF4.Dataset(g["meta"]["cbf"]) as ds:
        tix = np.array(ds["time"][:], dtype=float)
        skt_attr = (float(ds["SKT"].getncattr("reference_mean"))
                    if "reference_mean" in ds["SKT"].ncattrs() else None)
        skt = np.array(ds["SKT"][:], dtype=float)
        edc_eqf = 2.0  # CBF has no EDC_EQF variable -> C default (READ_NETCDF)

    cfg = {"n_timesteps": len(tix),
           "dint": edcs.compute_dint(tix),
           "edc_eqf": edc_eqf,
           "skt_ref_mean": edcs.skt_reference_mean(skt, skt_attr)}

    def one(pars, pools, fluxes):
        vals = edcs.edc_values(pars, pools, fluxes, cfg)
        rec_pre, p_pre = edcs.record_phase(vals, True, jnp.asarray(0.0))
        rec_post, _ = edcs.record_phase(vals, False, jnp.asarray(0.0))
        invoke_post = p_pre > -jnp.inf     # MLF2: EDC==1 & P>-INFINITY
        rec = rec_pre + jnp.where(invoke_post, rec_post,
                                  jnp.zeros(NOEDCS))
        return rec

    rec = jax.vmap(one)(g["params"], g["pools"], g["fluxes"])
    golden = g["mlf"][:, :NOEDCS]
    return np.asarray(rec), golden


def test_edc_records(edc_results):
    rec, golden = edc_results
    n = rec.shape[0]
    bad = []
    for i in range(n):
        for k in range(NOEDCS):
            c, j = golden[i, k], rec[i, k]
            if np.isnan(c) or np.isnan(j):
                ok = np.isnan(c) and np.isnan(j)
            elif np.isinf(c) or np.isinf(j):
                ok = c == j
            else:
                ok = abs(j - c) <= 1e-10 * max(abs(c), 1e-30) or \
                     abs(j - c) <= 1e-12
            if not ok:
                bad.append((i, k, c, j))
    assert not bad, f"{len(bad)} mismatches, first 10: {bad[:10]}"


def test_gate_masks_match(edc_results):
    """The short-circuit zero pattern (which slots were never written) must
    reproduce exactly — it encodes the recording semantics."""
    rec, golden = edc_results
    np.testing.assert_array_equal(rec == 0.0, golden == 0.0)
