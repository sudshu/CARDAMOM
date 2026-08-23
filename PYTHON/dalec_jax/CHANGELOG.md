# dalec_jax session log

Newest first. Every agentic session appends: what was attempted, what passed,
what FAILED (so it is not blindly retried), environment changes.

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
