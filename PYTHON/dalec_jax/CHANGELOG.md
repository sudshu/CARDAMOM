# dalec_jax session log

Newest first. Every agentic session appends: what was attempted, what passed,
what FAILED (so it is not blindly retried), environment changes.

## 2026-08-23 — Session 1 (P1, same session as P0)

- Oracle harness `C/projects/JAX_VALIDATION/oracle_1100.c` built (gcc 8.5.0,
  glibc 2.28, netCDF 4.7.0; flags = production: no -O). Subcommands:
  manifest / module / trajectory / mlf.
- **Validation milestone:** trajectory + mlf outputs are BIT-IDENTICAL to
  CARDAMOM_RUN_MODEL.exe for every one of the 3614/4000 posterior samples
  that genuinely run (the other 386 are prerun-EDC-gated burn-in whose
  RUN_MODEL output is stale — bug confirmed live: gated samples' RUN_MODEL
  trajectories are zeros/stale while the oracle shows the true divergent
  runs). Determinism: byte-identical across repeated runs.
- **Buffer-staleness discovery beyond the plan:** production callocs
  M_POOLS/M_FLUXES once per process, so a break-sample's tail holds the
  PREVIOUS sample's values (not zeros) for samples >0 — in-process C output
  is sample-order dependent. Oracle zeroes buffers per sample
  (BUG_COMPAT: per_sample_buffer_zeroing).
- indices.py GENERATED (gen_indices.py --check green). True index facts:
  F.ph_fol2lit=7, F.lab2lit=11, F.dist_lab=89 (the old port's dist_lab=7 and
  the exploration report's 85 were both wrong — parse, never trust).
- ULP census (tools/ulp_census.py, n=200k/case) → tests/TOLERANCES.md:
  XLA CPU x64 is bit-identical to glibc for log/sqrt/pow/sin/cos/tan;
  exp ≤2 ULP; acos ≤1 ULP; erfc ≤34 ULP / 4.1e-15 rel (sole real source);
  numpy exp/cos bit-identical to libm → VegK prederive in numpy is exact.
  L1 tolerances tightened accordingly (most modules target bit-exact).
- Goldens: 16 modules × ~4.1k cases (LHS 4096 + edge rows) + 120 trajectory
  fixtures (64 genuine / 16 gated / 8 viable / 32 prior; 53 break-path).
  34 MB, gitignored, deterministic (byte-compared regeneration).
- CBF obs coverage for likelihood tests: GPP 240, SCF 240, ABGB 228,
  LAI 192, EWT 192, ROFF 133 valid points.
- DEFERRED: ORACLE_TAP macros → P3.

## 2026-08-23 — Session 1 (P0)

- Forked CARDAMOM-framework/CARDAMOM → sudshu/CARDAMOM; created `jax-port`
  branch from `5ddb4b7b` (dalec-baseline HEAD) and pushed.
- Installed `jax[cuda12]==0.11.1` into the research repo's `.venv`
  (Python 3.13, numpy 2.5.2). Two idle A100-40GB available.
- Created package skeleton `PYTHON/dalec_jax/` (pyproject, src layout,
  smoke test) + state docs (CLAUDE.md rules, plan.md, BUG_COMPAT.md, this
  file).
- Downloaded 4/5 reference papers to research repo `papers/`
  (Bloom 2016 PNAS is bot-walled from this host — fetch manually; analyses
  are specified in the open sources meanwhile).
- Known upstream context: dalec-baseline's two prior commits are
  "Update DALEC_1100_JAX_MLF.py" — upstream is actively iterating a partial
  JAX port with known index/constant defects (see BUG_COMPAT.md and the
  research-repo plan); our port is clean-room with that file as reference
  only.
- **Environment gotcha (P0):** the login shell exports
  `LD_LIBRARY_PATH=/usr/local/cuda/lib64`, which shadows the pip-installed
  CUDA 12.9 wheels and breaks the jax GPU plugin ("cuSPARSE not found",
  CPU fallback). Run everything with `env -u LD_LIBRARY_PATH`. With it unset,
  both A100s enumerate and the full smoke suite passes (5/5, incl. GPU scan).
- Smoke gate GREEN: 5/5 pytest (x64 verified, scan+where+vmap pipeline,
  alive-flag freeze-to-zero semantics, f64 transcendentals, GPU x64 scan).
- Next (P1): oracle_1100.c subcommands, gen_indices/gen_fixtures, ULP census.
