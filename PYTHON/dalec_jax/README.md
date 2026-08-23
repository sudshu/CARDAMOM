# dalec_jax

A JAX port of CARDAMOM's **DALEC_1100** (forward model + the 15 live EDCs +
the DALEC_MLF2 likelihood) whose defining property is **verified numerical
equivalence with the C code**. The C under `../../C/` is run as an oracle
(`../../C/projects/JAX_VALIDATION/oracle_1100.c`) and the test suite asserts
the JAX implementation reproduces it.

## Verified equivalence (all gates green, 2026-08-23)

| Level | Scope | Result |
| --- | --- | --- |
| L1 | 16 leaf modules × ~4.1k golden cases | 11 modules **bit-identical**; daylength ≤4 ULP (jnp.arccos row); exp/erfc chains ≤1e-12 mixed (census-cited) |
| L4 | 120 full 240-step trajectories | 94 pointwise ≤1e-10 (break step + zero tail exact); 26 chaos-certified — divergence onset inside the C's **own** 1-ULP self-divergence envelope (K=8 parameter dithers through the C oracle) |
| L2/L5 | 15 EDCs × 4000 posterior samples | 60,000 recorded slots, **zero mismatches** (booleans, −inf/NaN sentinels, short-circuit masks exact) |
| L6 | 31 likelihood terms + total P | ≤1e-12 on 120 fixtures; gated-row zero patterns exact |
| L7 | full pipeline, **all 4000 posterior samples** | ≥90% tight (rel ≤1e-9); remainder both-rejected (P<−1e5 both engines) or dither-certified |
| L8 | paper analyses (residence times, allocation, CI envelopes, CH4/Rh partition, NBE seasonality, FluxVal skill) | **59/59 derived quantities agree ≤1e-10** between engines |

Key methodological findings (details in `tests/TOLERANCES.md`):
- `--xla_disable_hlo_passes=algsimp` is **mandatory** — XLA otherwise
  rewrites `x/c → x·(1/c)` and `(x/c1)·c2 → x·(c2/c1)`, breaking
  bit-identity with `-O0` C.
- With it, XLA CPU x64 is bit-identical to glibc for log/sqrt/pow/trig;
  only exp (≤2 ULP), arccos (≤1 ULP) and erfc (≤34 ULP) differ.
- DALEC_1100 is ULP-chaotic for a subset of parameter draws: the C diverges
  from itself under 1-ULP parameter perturbations. Equivalence claims are
  therefore certified against the C's measured self-sensitivity — the JAX
  port is statistically indistinguishable from a 1-ULP-perturbed C.
- C defects are reproduced deliberately and cataloged in `BUG_COMPAT.md`
  (q_ly1 overflow slot, 7-digit π, stale-output paths, OBSOPE index bugs).

## Why

The C forward model is fast (~0.2 ms/trajectory). What C cannot provide:
- **Gradients**: `jax.grad` of the log-posterior over all 89 parameters in
  one backward pass (vs 90 forward runs for finite differences) — the
  gateway to HMC/NUTS and variational inference.
- **Batched GPU execution**: A100 `vmap(4000)` runs the full pipeline at
  0.029 ms/trajectory (6.7× one C core; forward 7.7×), scaling with batch.

## Quick start

```bash
pip install -e ".[test]"
make -C ../../C/projects/JAX_VALIDATION            # build the oracle
python tools/gen_fixtures.py                       # generate goldens (~1 min)
env -u LD_LIBRARY_PATH pytest                      # 30 tests
```

Requires Python ≥3.11, jax pinned exact (see pyproject). `jax_enable_x64`
is switched on at import; the equivalence suite is CPU-only by policy and
sets `XLA_FLAGS=--xla_disable_hlo_passes=algsimp` via `tests/conftest.py`.

## Layout

```
src/dalec_jax/
  constants.py     GLOBAL_CONSTANTS.c literals (incl. 7-digit pi)
  indices.py       GENERATED from DALEC_1100_INDICES.c — never hand-edited
  modules/         1:1 ports of the C leaf modules
  model/           the 240-step lax.scan model (DALEC_1100.c:459-1142)
  edcs/            7 prerun + 8 postrun EDCs + short-circuit recording
  likelihood/      obs data prep, operators, 10 filter modes, MLF2 gates
  io/              common xarray schema (feeds the analysis layer)
tools/             index generator, fixture/golden generator, ULP census,
                   benchmark + gradient scan
tests/             oracle-driven equivalence suite (TOLERANCES.md = contract)
```

State docs for agentic development: `CLAUDE.md` (transcription rules),
`plan.md` (per-phase gates), `CHANGELOG.md` (session log), `BUG_COMPAT.md`.
Methodology follows arXiv:2606.07681.
