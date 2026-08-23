"""L1: every leaf module vs its C-oracle golden, at TOLERANCES.md classes.

Requires --xla_disable_hlo_passes=algsimp (set by conftest.py): with XLA's
algebraic simplifier active, div-by-constant and (x/c1)*c2 rewrites break
bit-identity for pure-arithmetic modules.

Tolerance classes (cite tests/TOLERANCES.md, census + fusion findings,
2026-08-23):
  exact — +,-,*,/, sqrt, log, pow, trig chains: bit-identical (0 ULP)
  acos  — one arccos (<=1 ULP, 92% bitident) scaled by 24/pi -> <=4 ULP
  mixed(T) — exp/erfc-seeded chains where cancellation amplifies ULP:
      per-element |got-ref| / max(|ref|, RMS_col) <= T, non-finite patterns
      must match exactly. T cites the measured amplification table.
"""
import jax
import numpy as np
import pytest

import dalec_jax  # noqa: F401
from dalec_jax import oracle_io
from dalec_jax.modules import ORACLE_REGISTRY

EXACT = "exact"
ACOS = "acos"

TOL_CLASS = {
    "HYDROFUN_EWT2MOI": EXACT,
    "HYDROFUN_MOI2EWT": EXACT,
    "HYDROFUN_MOI2CON": EXACT,
    "HYDROFUN_MOI2PSI": EXACT,
    "HYDROFUN_PSI2MOI": EXACT,
    "DRAINAGE": EXACT,
    "INTERNAL_ENERGY_PER_LIQUID_H2O_UNIT_MASS": EXACT,
    "INITIALIZE_INTERNAL_SOIL_ENERGY": EXACT,
    "MIN_QUADRATIC_SMOOTH": EXACT,
    "SOIL_TEMP_AND_LIQUID_FRAC": EXACT,
    "HET_RESP_RATES_JCR": EXACT,
    "COMPUTE_DAYLIGHT_HOURS": ACOS,
    # mixed-criterion thresholds, from the measured amplification table:
    # ALLOC max 1.0e-14, KNORR max 1.7e-14 (erfc tails + dlambdadt
    # cancellation), LIU max 2.6e-13 (2-ULP exp through the (co2-ci)
    # conditioning in gs/transp). Bounds set ~5x above measurement.
    "MAX_EXPONENTIAL_SMOOTH": ("mixed", 5e-14),
    "ALLOC_AND_AUTO_RESP_FLUXES": ("mixed", 1e-13),
    "KNORR_ALLOCATION": ("mixed", 1e-13),
    "LIU_AN_ET": ("mixed", 1e-12),
}

_missing = set(TOL_CLASS) ^ set(ORACLE_REGISTRY)
assert not _missing, f"registry/tolerance mismatch: {_missing}"


def _run_jax(name: str, x: np.ndarray) -> np.ndarray:
    fn = ORACLE_REGISTRY[name]
    cols = [x[:, i] for i in range(x.shape[1])]
    out = jax.vmap(fn)(*[np.asarray(c) for c in cols])
    if isinstance(out, tuple):
        return np.column_stack([np.asarray(o) for o in out])
    return np.asarray(out)[:, None]


def _mixed_err(ref: np.ndarray, got: np.ndarray) -> np.ndarray:
    """Per-element |got-ref| / max(|ref|, column RMS); non-finite mismatch=inf."""
    out = np.zeros_like(ref)
    for c in range(ref.shape[1]):
        r, g = ref[:, c], got[:, c]
        fin = np.isfinite(r)
        rms = np.sqrt(np.mean(r[fin] ** 2)) if fin.any() else 1.0
        e = np.abs(g - r) / np.maximum(np.abs(r), max(rms, 1e-300))
        same_nonfin = (~np.isfinite(r) & ~np.isfinite(g)
                       & (np.isnan(r) == np.isnan(g))
                       & (np.isnan(r) | (np.sign(r) == np.sign(g))))
        out[:, c] = np.where(fin & np.isfinite(g), e,
                             np.where(same_nonfin, 0.0, np.inf))
    return out


@pytest.mark.parametrize("name", sorted(TOL_CLASS))
def test_module_equivalence(name):
    if not (oracle_io.GOLDEN / f"modules/{name}.in.bin").exists():
        pytest.skip("goldens missing -- run tools/gen_fixtures.py")
    x, ref = oracle_io.module_golden(name)
    got = _run_jax(name, x)
    assert got.shape == ref.shape

    spec = oracle_io.module_spec(name)
    cls = TOL_CLASS[name]
    if cls in (EXACT, ACOS):
        ulp = oracle_io.ulp_distance(ref, got)
        worst = ulp.max(axis=0)
        detail = "; ".join(f"{o}: ulp={int(u)}"
                           for o, u in zip(spec["outputs"], worst))
        limit = 0 if cls == EXACT else 4
        assert ulp.max() <= limit, f"{name} exceeds {limit} ULP ({detail})"
    else:
        _, tol = cls
        err = _mixed_err(ref, got)
        worst = err.max(axis=0)
        detail = "; ".join(f"{o}: mixed={e:.3g}"
                           for o, e in zip(spec["outputs"], worst))
        assert err.max() <= tol, f"{name} exceeds mixed {tol:g} ({detail})"
