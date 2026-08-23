# Equivalence tolerances — empirical basis (ULP census)

Every L1 tolerance in the test suite must cite a row of this census.
Regenerate with `env -u LD_LIBRARY_PATH JAX_PLATFORMS=cpu python
tools/ulp_census.py` after ANY change to glibc, gcc, or jax.

## Census (2026-08-23)

Environment: jax 0.11.1 CPU x64, numpy 2.5.2, glibc 2.28, gcc 8.5.0,
RHEL 8.10, n = 200,000 per case, seed 20260823. Reference = glibc libm via
ctypes (exactly what the -O0 C binaries call).

| case | max ULP | p99.9 ULP | max rel err | bit-identical |
| --- | ---: | ---: | ---: | ---: |
| exp [-45,45] | 2 | 1 | 3.07e-16 | 86.28% |
| log [1e-12,1e9] | 0 | 0 | 0 | 100% |
| sqrt [1e-12,1e12] | 0 | 0 | 0 | 100% |
| erfc [-8,8] | 34 | 29 | 4.08e-15 | 56.44% |
| pow q10 [1,6]^[-30,30] | 0 | 0 | 0 | 100% |
| pow retention (1e-4,1.3]^[1,30] | 0 | 0 | 0 | 100% |
| pow generic [1e-6,1e4]^[-4,4] | 0 | 0 | 0 | 100% |
| cos, sin, tan [-7,7] | 0 | 0 | 0 | 100% |
| acos [-1,1] | 1 | 1 | 2.22e-16 | 92.37% |
| numpy.exp vs libm (info) | 0 | 0 | 0 | 100% |
| numpy.cos vs libm (info) | 0 | 0 | 0 | 100% |

Reading: on this host XLA CPU lowers log/sqrt/pow/trig to libm → bit-identical
with the C. Only exp (≤2 ULP), acos (≤1 ULP) and erfc (≤34 ULP, ≤4.1e-15 rel)
can differ. numpy calls libm for exp/cos → the VegK solar-geometry prederive
computed in numpy is bit-identical to the C prederive.

## Derived per-level tolerances (supersede the provisional plan numbers)

| Level | Modules / scope | Assertion |
| --- | --- | --- |
| L1-exact | HYDROFUN×5, DRAINAGE, SOIL_TEMP_AND_LIQUID_FRAC, INTERNAL_ENERGY_*, INITIALIZE_INTERNAL_SOIL_ENERGY, MIN_QUADRATIC_SMOOTH (pow/sqrt only), HET_RESP_RATES_JCR (pow only), COMPUTE_DAYLIGHT_HOURS (trig+acos: allow ≤1 ULP) | bit-identical (≤1 ULP where acos present) |
| L1-exp | MAX_EXPONENTIAL_SMOOTH, ALLOC_AND_AUTO_RESP_FLUXES, LIU_AN_ET (exp/pow/sqrt chains) | ≤ 8 ULP per output, and rel ≤ 1e-14 |
| L1-erfc | KNORR_ALLOCATION (2 erfc calls in path) | rel ≤ 1e-13 |
| L3 | per-sub-block checkpoints, one step | rel ≤ 1e-13 |
| L4 | 240-step trajectories | rel ≤ 1e-10 vs max(|c|, RMS_var); break index exact; zero tail exact |
| L5 | postrun EDC booleans / stats | exact / rel ≤ 1e-10 |
| L6 | 31 likelihood terms, total P | rel ≤ 1e-12; sentinels exact |
| L7 | 4000-sample posterior sweep | gates exact; worst L4 criterion; P rel ≤ 1e-12 |
| L8 | paper-analysis outputs | rel ≤ 1e-10 |

## Fusion findings (2026-08-23, L1 bring-up)

The census measured UNFUSED single-op dispatch. Under jit, two additional
divergence sources appeared and were root-caused:

1. **XLA algebraic simplifier rewrites arithmetic** — confirmed in optimized
   HLO: `x / const` → `x * (1/const)` and `(x/c1) * c2` → `x * (c2/c1)`
   (INITIALIZE_INTERNAL_SOIL_ENERGY kernel showed both, 1–2 ULP each).
   RESOLUTION: `--xla_disable_hlo_passes=algsimp` is MANDATORY for all
   equivalence runs (enforced by tests/conftest.py before jax import).
   With it, all pure-arithmetic/pow/log/sqrt/trig modules are bit-identical
   to the C oracle over the full Tier-A fixture sets.
2. **jnp.arccos is 1 ULP off glibc on ~3% of arguments** (census row);
   scaled by 24/pi in ComputeDaylightHours → ≤4 ULP final. Class "acos" = 4.

Measured worst-case mixed error (|Δ| / max(|ref|, column RMS)) for the
exp/erfc chains at 4.1k Tier-A cases, with algsimp disabled:

| module | worst output | measured | bound set |
| --- | --- | ---: | ---: |
| MAX_EXPONENTIAL_SMOOTH | maxx | ~1e-15 | 5e-14 |
| ALLOC_AND_AUTO_RESP_FLUXES | ALLOC_FOL_ACTUAL | 1.0e-14 | 1e-13 |
| KNORR_ALLOCATION | f_T (erfc tail) | 1.7e-14 | 1e-13 |
| LIU_AN_ET | transp | 2.6e-13 | 1e-12 |

LIU's amplification is the ≤2-ULP exp error propagated through the
(co2−ci) conditioning into gs/transp — arithmetic is bit-exact, the seed is
exp alone. Huge raw-ULP counts on outputs approaching 0 (denormal
LEAF/NONLEAF mortality factors, |Δ| ≤ 1e-290) are absorbed by the RMS floor
and are physically meaningless.

Notes.
- Equivalence tests are CPU-only by policy (GPU runs are for performance,
  not reference comparison); conftest sets JAX_PLATFORMS=cpu.
- erfc error budget over a trajectory: ≤4.1e-15 rel per step into the
  phenology memory state; linear accumulation over 240 steps ≈ 1e-12,
  two orders inside the 1e-10 L4 bar. The fdlibm-erfc escape hatch stays
  pre-authorized but is not expected to trigger.
- Any override beyond these values requires a new census row plus a
  BUG_COMPAT or CHANGELOG entry explaining the op-level cause.
