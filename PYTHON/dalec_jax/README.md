# dalec_jax

A JAX port of CARDAMOM's **DALEC_1100** model (forward model + the 15 live
EDCs + the DALEC_MLF2 likelihood) whose defining property is **verified
numerical equivalence with the C code**: the C under `../../C/` is run as an
oracle (`../../C/projects/JAX_VALIDATION/`) to produce golden reference
outputs, and the test suite asserts the JAX implementation reproduces them —
leaf modules to ~1e-15/1e-13 relative, full 240-step trajectories to 1e-10
per timestep, EDC pass/fail booleans and likelihood sentinels exactly.

Why: DALEC_1100 in C is already fast (~0.2 ms per trajectory). What C cannot
provide is **gradients** (HMC/NUTS, variational inference, optimization) and
**batched GPU execution** (`vmap` over thousands of parameter vectors). This
port provides both without changing the science — bit-compatibility with the
C, including its documented quirks (see `BUG_COMPAT.md`), is a test-enforced
invariant.

Status: under construction on the `jax-port` branch — see `plan.md` for
per-module progress and `CHANGELOG.md` for the session log. Conversion
methodology follows arXiv:2606.07681 (*Systematic LLM Translation of Legacy
Scientific Code to Differentiable Frameworks*).

## Layout

```
src/dalec_jax/
  constants.py     transcribed GLOBAL_CONSTANTS.c literals (incl. 7-digit pi)
  indices.py       GENERATED from DALEC_1100_INDICES.c — never hand-edited
  modules/         1:1 ports of the C leaf modules (LIU_AN_ET, KNORR, ...)
  model/           the 240-step lax.scan model (DALEC_1100.c:459-1142)
  edcs/            7 prerun + 8 postrun EDCs + dispatcher gate
  likelihood/      29 obs operators, 10 opt_filter modes, MLF2 composition
  io/              CBF/CBR readers, common xarray output schema
tools/             index generator, fixture generator
tests/             oracle-driven equivalence suite (L0–L8; see TOLERANCES.md)
```

## Quick start (development)

```bash
pip install -e ".[test]"          # jax pinned exact — see pyproject.toml
make -C ../../C/projects/JAX_VALIDATION golden   # build oracle + goldens
pytest
```

Requires Python ≥ 3.11. `jax_enable_x64` is switched on at import; float32
use anywhere is a defect (`(D_LF_LY1 + D_LF_LY2) == 2` in the C is an exact
float64 comparison the port must honor).
