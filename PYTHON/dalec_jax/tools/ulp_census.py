#!/usr/bin/env python3
"""Transcendental ULP census: glibc libm (what -O0 C calls) vs jax.numpy.

The C reference and the JAX port agree bit-for-bit on +,-,*,/ and sqrt given
identical operation order (IEEE 754 exact operations). Divergence can only
enter through transcendentals. This script measures, for each transcendental
the model uses, the worst-case ULP distance between glibc's double routine
(called through ctypes, exactly what the -O0 C binary calls) and the jnp
equivalent on CPU x64, over the argument ranges the model actually exercises.

Output feeds tests/TOLERANCES.md; every L1 tolerance override must cite a row
from this census. Rerun whenever glibc or jax is upgraded.

Usage: env -u LD_LIBRARY_PATH python tools/ulp_census.py [--n 200000]
"""
from __future__ import annotations

import argparse
import ctypes
import ctypes.util

import numpy as np

import dalec_jax  # noqa: F401  -- enables x64 before jnp use

import jax
import jax.numpy as jnp
import jax.scipy.special as jsp

LIBM = ctypes.CDLL(ctypes.util.find_library("m"))
for fname in ("exp", "log", "sqrt", "erfc", "cos", "sin", "tan", "acos", "asin"):
    f = getattr(LIBM, fname)
    f.restype = ctypes.c_double
    f.argtypes = [ctypes.c_double]
LIBM.pow.restype = ctypes.c_double
LIBM.pow.argtypes = [ctypes.c_double, ctypes.c_double]


def libm1(name: str, x: np.ndarray) -> np.ndarray:
    f = getattr(LIBM, name)
    return np.array([f(float(v)) for v in x])


def libm_pow(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.array([LIBM.pow(float(a), float(b)) for a, b in zip(x, y)])


def ulp_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """ULP distance between two float64 arrays (finite entries only)."""
    ia = a.view(np.int64).copy()
    ib = b.view(np.int64).copy()
    # map to monotonic integer line (two's-complement trick for negatives)
    ia = np.where(ia < 0, np.int64(-(2**63)) - ia - 1, ia)
    ib = np.where(ib < 0, np.int64(-(2**63)) - ib - 1, ib)
    d = np.abs(ia - ib)
    both_fin = np.isfinite(a) & np.isfinite(b)
    same_nonfin = ~np.isfinite(a) & ~np.isfinite(b) & (
        np.isnan(a) == np.isnan(b))
    d = np.where(both_fin, d, np.where(same_nonfin, 0, np.int64(2**62)))
    return d


def rel_err(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(a) & np.isfinite(b) & (np.abs(a) > 0)
    if not m.any():
        return 0.0
    return float(np.max(np.abs(a[m] - b[m]) / np.abs(a[m])))


# (name, sampler, libm evaluator, jnp evaluator, model usage note)
def build_cases(rng: np.random.Generator, n: int):
    def logu(lo, hi, size):
        return np.exp(rng.uniform(np.log(lo), np.log(hi), size))

    x_exp = rng.uniform(-45.0, 45.0, n)
    x_log = logu(1e-12, 1e9, n)
    x_sqrt = logu(1e-12, 1e12, n)
    x_erfc = rng.uniform(-8.0, 8.0, n)
    # pow classes the model uses:
    #   Q10-style: base in [1,6], exponent in [-30, 30]
    b_q10 = rng.uniform(1.0, 6.0, n)
    e_q10 = rng.uniform(-30.0, 30.0, n)
    #   soil-retention: base = moi in (1e-4, 1.3), exponent = b-ish in [1, 30]
    b_ret = logu(1e-4, 1.3, n)
    e_ret = rng.uniform(1.0, 30.0, n)
    #   squares/quartics with fractional bases
    b_sq = logu(1e-6, 1e4, n)
    e_sq = rng.uniform(-4.0, 4.0, n)
    # trig for VegK/daylength prederive (numpy path, still censused)
    x_trig = rng.uniform(-7.0, 7.0, n)
    x_inv = rng.uniform(-1.0, 1.0, n)

    return [
        ("exp [-45,45]", x_exp, lambda x: libm1("exp", x),
         lambda x: np.asarray(jnp.exp(jnp.asarray(x)))),
        ("log [1e-12,1e9]", x_log, lambda x: libm1("log", x),
         lambda x: np.asarray(jnp.log(jnp.asarray(x)))),
        ("sqrt [1e-12,1e12]", x_sqrt, lambda x: libm1("sqrt", x),
         lambda x: np.asarray(jnp.sqrt(jnp.asarray(x)))),
        ("erfc [-8,8]", x_erfc, lambda x: libm1("erfc", x),
         lambda x: np.asarray(jsp.erfc(jnp.asarray(x)))),
        ("pow q10 [1,6]^[-30,30]", (b_q10, e_q10),
         lambda t: libm_pow(*t),
         lambda t: np.asarray(jnp.power(jnp.asarray(t[0]), jnp.asarray(t[1])))),
        ("pow retention (1e-4,1.3]^[1,30]", (b_ret, e_ret),
         lambda t: libm_pow(*t),
         lambda t: np.asarray(jnp.power(jnp.asarray(t[0]), jnp.asarray(t[1])))),
        ("pow generic [1e-6,1e4]^[-4,4]", (b_sq, e_sq),
         lambda t: libm_pow(*t),
         lambda t: np.asarray(jnp.power(jnp.asarray(t[0]), jnp.asarray(t[1])))),
        ("cos [-7,7]", x_trig, lambda x: libm1("cos", x),
         lambda x: np.asarray(jnp.cos(jnp.asarray(x)))),
        ("sin [-7,7]", x_trig, lambda x: libm1("sin", x),
         lambda x: np.asarray(jnp.sin(jnp.asarray(x)))),
        ("tan [-7,7]", x_trig, lambda x: libm1("tan", x),
         lambda x: np.asarray(jnp.tan(jnp.asarray(x)))),
        ("acos [-1,1]", x_inv, lambda x: libm1("acos", x),
         lambda x: np.asarray(jnp.arccos(jnp.asarray(x)))),
        ("numpy.exp vs libm (info)", x_exp, lambda x: libm1("exp", x),
         lambda x: np.exp(x)),
        ("numpy.cos vs libm (info)", x_trig, lambda x: libm1("cos", x),
         lambda x: np.cos(x)),
    ]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=20260823)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    dev = jax.devices()[0].platform
    print(f"jax {jax.__version__} on {dev}; numpy {np.__version__}; "
          f"n={a.n} seed={a.seed}")
    print(f"{'case':38s} {'max ULP':>10s} {'p99.9 ULP':>10s} "
          f"{'max rel err':>12s} {'bitident %':>10s}")
    for name, x, fc, fj in build_cases(rng, a.n):
        ref = fc(x)
        got = fj(x)
        d = ulp_dist(ref, got)
        bit = 100.0 * float(np.mean(d == 0))
        print(f"{name:38s} {int(d.max()):>10d} "
              f"{int(np.percentile(d, 99.9)):>10d} {rel_err(ref, got):>12.3e} "
              f"{bit:>9.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
