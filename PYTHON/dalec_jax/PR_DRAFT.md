# PR draft — dalec_jax: an oracle-verified JAX port of DALEC_1100

Target: `CARDAMOM-framework/CARDAMOM` ← `sudshu/CARDAMOM:jax-port`
(NOT yet opened — awaiting maintainer contact / owner approval.)

---

**Title:** dalec_jax: JAX port of DALEC_1100 with C-oracle-verified numerical
equivalence (forward model + EDCs + likelihood)

## What this adds

- `PYTHON/dalec_jax/` — a pip-installable JAX implementation of DALEC_1100:
  forward model (`lax.scan`), all 15 live EDCs, and the DALEC_MLF2
  likelihood (29 observation operators, all 10 opt_filter modes).
- `C/projects/JAX_VALIDATION/oracle_1100.c` — a golden-reference harness
  that batch-evaluates any leaf module, runs `DALEC_1100()` directly
  (bypassing the DALEC_MLF2 stale-output path), and dumps
  EDCs/likelihoods/P per parameter vector. Compiled with the production
  flags (plain gcc, no -O).
- A 30-test equivalence suite with an explicit tolerance contract
  (`tests/TOLERANCES.md`) and a bug-compatibility register
  (`BUG_COMPAT.md`).

## Why

Gradients and batched GPU execution. `jax.grad` gives dP/d(all 89
parameters) in one backward pass — the door to HMC/NUTS instead of
random-walk MHMCMC — and an A100 runs the full pipeline for 4000 parameter
vectors at 0.029 ms/trajectory (6.7× one C core).

## How equivalence is demonstrated (headline numbers)

- 11 of 16 leaf modules are **bit-identical** to the C over ~4,100
  Latin-hypercube + branch-edge cases each (requires disabling XLA's
  `algsimp` pass, which otherwise rewrites division arithmetic).
- Full 240-step trajectories: 94/120 fixtures pointwise ≤1e-10 with exact
  break-step/zero-tail semantics; the remaining 26 diverge **only** where
  the C itself diverges under 1-ULP parameter perturbations (measured with
  K=8 dither runs of the C oracle — the JAX onsets sit inside the C's own
  self-divergence envelope).
- EDCs: 60,000 recorded slots across the 4000-sample posterior — zero
  mismatches, including −inf sentinels and the dispatcher's short-circuit
  masks.
- Likelihood: 31 terms ≤1e-12; full-pipeline P over all 4000 posterior
  samples ≥90% at rel ≤1e-9, remainder accounted (both-rejected or
  chaos-certified).
- Paper-style analyses (residence times, allocation fractions, CI
  envelopes, CH4/Rh partition, NBE seasonality, FluxVal-style skill): 59/59
  derived quantities agree ≤1e-10 between engines.

## Findings upstream may care about (all documented in BUG_COMPAT.md)

1. `CARDAMOM_RUN_MODEL.exe` writes **stale trajectories** for
   prerun-EDC-failing samples (`DALEC_MLF2.c:47` skips the model; buffers
   are process-lifetime). 386/4000 samples of our test posterior are
   affected in its output.
2. LY2 overflow accumulates into `q_ly1` (`DALEC_1100.c:784`) — per-layer
   runoff split is wrong, ROFF total unaffected.
3. `OBSOPE.rhch4_rhco2_flux = F.rh_ch4/F.rh_co2` (integer division of two
   flux indices) is then used as a **PARS** index; `C3frac_PARAM` is never
   assigned (reads pars[0]).
4. `READ_NETCDF_SINGLE_OBS_FIELDS` reads uninitialized
   `min_value`/`max_value` when setting `validobs` (no ML effect — the
   likelihood gates on `value != -9999`).
5. The existing `PYTHON/.../DALEC_1100_JAX_MLF.py` has misassigned flux
   indices (e.g. `dist_lab=7` collides with `ph_fol2lit`; true value 89),
   a non-C drainage function, and non-C constants — this port supersedes
   it as a validated implementation (kept untouched in this PR).

## Not in scope

MCMC reimplementation (the C samplers remain authoritative), gradient
hardening of the where-branch sites (catalogued; follow-on), multi-model
generalization beyond ID 1100.

🤖 Generated with [Claude Code](https://claude.com/claude-code)

https://claude.ai/code/session_01EtMWeqbawAK9mUR4GH8rQ1
