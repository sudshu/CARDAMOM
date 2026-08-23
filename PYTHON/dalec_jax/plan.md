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

## P2 — leaf modules (gate: L1 green at census tolerances) — DONE 2026-08-23
- [x] All 16 oracle modules ported and green (20 passed):
      11 BIT-IDENTICAL (hydrofun×5, drainage, soil_temp, soil_energy×2,
      min_quad_smooth, het_resp_rates_jcr), daylength ≤4 ULP (arccos row),
      mixed-criterion: max_exp_smooth 5e-14, alloc 1e-13, knorr 1e-13,
      liu 1e-12 (measured 2.6e-13 worst, exp→(co2−ci) conditioning).
- [x] MANDATORY FLAG DISCOVERED: --xla_disable_hlo_passes=algsimp
      (XLA rewrites x/c→x·(1/c), (x/c1)·c2→x·(c2/c1)); enforced in
      tests/conftest.py before jax import. See TOLERANCES.md "Fusion findings".

## P3 — step body + trajectory — DONE 2026-08-23 (gate redefined, see TOLERANCES.md "Trajectory chaos")
- [x] model/dalec_1100.py — VegK prederive (numpy), init block, literal
      680-line step transcription (4-pass updates, q_ly1 overflow bug
      reproduced, post-overflow q ordering into energy fluxes), lax.scan +
      alive-flag freeze-to-zero
- [x] L4 GREEN on all 120 fixtures: 94 pointwise-clean (≤1e-10 mixed or
      ≤1e-12 abs, break step + zero tail exact), 26 chaos-certified
      (divergence onset within the C's own 1-ULP self-divergence envelope,
      measured by K=8 all-parameter dither runs of the C oracle)
- [x] Chaos certification integrated into gen_fixtures.py
      (trajectories/chaos_cert.json, same criterion code as the test)
- [~] L3 tap checkpoints SUPERSEDED (rationale in TOLERANCES.md); ORACLE_TAP
      design shelved as future debugging tool
- [ ] types.py PyTrees — deferred to P5 (only needed for likelihood config)

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
