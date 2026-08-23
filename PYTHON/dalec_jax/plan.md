# dalec_jax port plan — per-module status

Legend: `[ ]` not started · `[~]` in progress · `[x]` done (test gate green) ·
`[!]` blocked (see CHANGELOG). Phases and gates from the approved plan
(research repo: ~/.claude/plans/i-would-like-to-proud-bumblebee.md).

## P0 — environment + repo (gate: smoke pytest; C builds pinned)
- [x] Fork sudshu/CARDAMOM, branch jax-port @ 5ddb4b7b, pushed
- [x] jax 0.11.1 (cuda12) in research-repo .venv, x64 smoke on CPU + A100
- [x] Package skeleton (pyproject, src layout, tests/test_smoke.py)
- [x] State docs: CLAUDE.md, plan.md, CHANGELOG.md, BUG_COMPAT.md
- [x] Papers downloaded (4/5; Bloom 2016 PNAS bot-walled — see research repo papers/README.md)

## P1 — oracle + codegen + census (gate: goldens bit-identical ×2; counts 89/100/30/15; TOLERANCES.md)
- [x] C/projects/JAX_VALIDATION/oracle_1100.c — `manifest` subcommand
- [x] `module` subcommand + adapters (16 modules registered)
- [x] `trajectory` subcommand — VALIDATED bit-identical vs RUN_MODEL on all
      3614/4000 genuinely-run posterior samples
- [x] `mlf` subcommand — P bit-identical vs RUN_MODEL PROB on all 3614
- [~] ORACLE_TAP macro lines in DALEC_1100.c — DEFERRED to P3 (checkpoint tests)
- [x] Makefile (pinned production flags gcc -O0) + fingerprint target
- [x] tools/gen_indices.py → indices.py (89/100/30/15 exact; --check mode)
- [x] tools/gen_fixtures.py + module_ranges.yaml (LHS 4096 + edges; 16 modules;
      determinism gate byte-identical)
- [x] Trajectory fixtures: 120 rows = 64 genuine + 16 gated + 8 viable +
      32 prior (53 exercise the isfinite-break path)
- [x] ULP census → tests/TOLERANCES.md. Headline: log/sqrt/pow/trig BIT-IDENTICAL
      to glibc; exp ≤2 ULP; erfc ≤34 ULP (4.1e-15 rel) — only real divergence.
      L1 targets tightened: most modules now target bit-exact.

## P2 — leaf modules (gate: L1 green at census tolerances)
Port order (dependency-sorted, LIU last):
- [ ] modules/hydrofun.py — EWT2MOI, MOI2EWT, MOI2CON, MOI2PSI, PSI2MOI
- [ ] modules/soil_energy.py — INTERNAL_ENERGY_PER_LIQUID_H2O_UNIT_MASS, INITIALIZE_INTERNAL_SOIL_ENERGY
- [ ] modules/soil_temp_liquid_frac.py — SOIL_TEMP_AND_LIQUID_FRAC
- [ ] modules/drainage.py — DRAINAGE
- [ ] modules/het_resp_rates_jcr.py — HET_RESP_RATES_JCR
- [ ] modules/lai_knorr_funcs.py — MinQuadraticSmooth, MaxExponentialSmooth, ComputeDaylightHours
- [ ] modules/knorr_allocation.py — KNORR_ALLOCATION (erfc — watch census)
- [ ] modules/alloc_and_auto_resp.py — ALLOC_AND_AUTO_RESP_FLUXES
- [ ] modules/liu_an_et.py — LIU_AN_ET_REFACTOR (largest, 264 lines C)

## P3 — step body + trajectory (gate: L3 ≤ 1e-13; L4 ≤ 1e-10 on 104 fixtures incl. exact break indices)
- [ ] types.py PyTrees (carry, params, forcing, config)
- [ ] model/dalec_1100.py — VegK prederive (numpy, libm-exact), init block
- [ ] Step body: literal transcription of D1100.c:459-1142 (sub-block order preserved; 4-pass updates)
- [ ] lax.scan wrapper + alive-flag freeze-to-zero
- [ ] tests/test_checkpoints.py (L3), tests/test_trajectory.py (L4)

## P4 — EDCs (gate: L2+L5 booleans exact on 4000 posterior + 10k prior)
- [ ] edcs/prerun.py — 7 parameter-only EDCs
- [ ] edcs/postrun.py — state_ranges, state_trajectories (C int-division stride), nsc_ratio, cfcr_ratio, fffr_ratio, mean_ly{1,2,3}_temp
- [ ] edcs/dispatcher.py — gate arithmetic incl. short-circuit staleness reproduction

## P5 — likelihood + posterior E2E (gate: L6+L7 green)
- [ ] likelihood/data_prep.py — TIMESERIES_OBS_STRUCT_PREPROCESS parity (numpy, load time)
- [ ] likelihood/obs_operators.py — 29 operators
- [ ] likelihood/timeseries_likelihood.py — 10 opt_filter modes (static indices), 3 unc types, 3 normalizations
- [ ] likelihood/single_obs_likelihood.py
- [ ] likelihood/mlf2.py — DALEC_MLF2 composition (prerun gate → model → postrun → likelihood)
- [ ] tests/test_posterior_e2e.py — all 4000 posterior samples vs `mlf`/`trajectory` oracle

## P6 — performance + gradient hygiene (gate: no NaN/Inf grads at 64 posterior points)
- [ ] vmap(4000) benchmark vs 0.196 ms/traj C — CPU + A100 report
- [ ] jax.grad NaN scan over the where-site audit list

## P7 — analyses + docs + PR (gate: L8 equality; PR opened)
- [ ] io/to_xarray.py common schema; analysis layer in research repo
- [ ] Residence times, allocation fractions, CI envelopes, CH4 + aerobic:anaerobic split, NBE seasonal cycle, FluxVal skill stats — run from BOTH engines, assert ≤1e-10
- [ ] Figure pack; README; BUG_COMPAT.md final; upstream PR draft
