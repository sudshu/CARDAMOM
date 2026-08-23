# dalec_jax — conventions and transcription rules

JAX port of DALEC_1100 (+ EDCs + DALEC_MLF2 likelihood) with C-oracle-verified
numerical equivalence. The C source under `../../C/` is the single source of
truth for behaviour; this document is the single source of truth for HOW to
transcribe it. Read `plan.md` for per-module status and `CHANGELOG.md` for
session history before starting work. Methodology follows arXiv:2606.07681.

## Equivalence bar (locked by user)

- `jax.config.update("jax_enable_x64", True)` before anything else — enforced
  in `src/dalec_jax/__init__.py`; never construct float32 arrays.
- Full-trajectory (240 steps): per-timestep per-variable
  `abs(jax - c) <= 1e-10 * max(abs(c), RMS_var)`.
- Leaf modules: arithmetic-only 1e-15; exp/log/pow 5e-14; erfc 1e-13 —
  provisional until the ULP census in `tests/TOLERANCES.md`; any override must
  cite a census row.
- EDC pass/fail booleans and −inf sentinels: exact.
- Likelihood terms: 1e-12 relative.

## Transcription rules (violations are review-blockers)

1. **Preserve C operation order.** No algebraic simplification, no hoisting,
   no reassociation. `sum(x_i / c)` stays a per-term division — do NOT rewrite
   as `sum(x_i) / c`.
2. **Sequential pool-update passes stay sequential.** DALEC_1100.c:964-1069
   updates pools in 4 passes (disturbance → fire → aggregate mortality →
   background turnover); each pass reads the previous pass's writes. Transcribe
   as 4 chained pure updates on the carry, never one fused update.
3. **Reductions that must match C bit-for-bit** (EDC statistics near
   thresholds): use `lax.scan` sequential accumulation, not `jnp.sum`, when the
   L5 escalation path is triggered (see TOLERANCES.md).
4. **Every division, log, pow, and sqrt gets a both-branches-safe guard.**
   `jnp.where(cond, a/b, c)` evaluates `a/b` even when `cond` is False; use the
   double-where idiom:
   `safe = jnp.where(cond, b, 1.0); out = jnp.where(cond, a/safe, c)`.
   Known mandatory sites: slf/SWE (D1100:588), DMF/TotalABGB (:979), 1/gs
   (LIU:233), moi2psi `pow(1/moi, b)`, trajectory-EDC `log(Fin/Fout)`.
5. **C `&`/`|` on comparisons are logical** — translate to
   `jnp.logical_and/or`, never Python `and/or` (traced) and never bitwise on
   floats.
6. **`-INFINITY` sentinels are semantics, not errors.** EDC and likelihood
   composition uses `P > -inf` gates; reproduce the arithmetic exactly. Under
   jit the model always runs — apply the gate as a mask afterwards.
7. **C integer division is truncation.** Where the C computes strides/bounds
   with int division (DALEC_EDC_TRAJECTORY.c:83-90), compute them in Python
   ints with `//` on the already-truncated operands, outside the trace.
8. **isfinite-break semantics** (D1100:1137-1141): the breaking step's
   non-finite values ARE written; all later steps are ZERO (calloc), not NaN.
   In the scan, carry an `alive` flag: record the poisoned step, then write
   zeros while `alive` is False. Break-step index must match C exactly.
9. **Struct `.IN` fields refilled per step vs set once** — every module call
   site passes the FULL input tuple explicitly; consult the C call site for
   which fields are loop-invariant. Never rely on Python closure state.
10. **Constants come from `constants.py` only**, transcribed literally from
    `GLOBAL_CONSTANTS.c`: `DGCM_PI = 3.1415927` (7 digits — do NOT use
    math.pi), `c_water = 4186.`, `c_ice = 2093.`, etc.
11. **Indices come from the generated `indices.py` only** (regenerated from
    `DALEC_1100_INDICES.c` by `tools/gen_indices.py`; a test diffs a fresh
    regeneration). Hand-writing an index literal anywhere is a defect — this
    exact failure mode (dist_lab=7 vs ph_fol2lit=7) sank the earlier
    `DALEC_1100_JAX_MLF.py` port.

## Bug-compatibility policy

The port reproduces C behaviour bit-for-bit including known defects. Every
such site is registered in `bug_compat.py` + `BUG_COMPAT.md` and marked in
code with `# BUG_COMPAT: <id>`. Never silently fix; never add a non-C clamp
or epsilon "for safety" (that's how the old port diverged). Scientific fixes
(e.g. capping the Rh temperature scalar) are out of scope and require
explicit user approval.

## Oracle workflow

- Build: `make -C ../../C/projects/JAX_VALIDATION` (pins production flags:
  gcc -O0, -lm, nc-config flags — matching BASH/CARDAMOM_COMPILE.sh).
- `oracle_1100 manifest` emits module I/O field order as JSON — the fixture
  generator and pytest loader both consume it; never hand-code field order.
- Goldens are regenerated (`make golden`), gitignored except the committed
  smoke subset; every golden set embeds an environment fingerprint (C git SHA,
  gcc/glibc versions, CFLAGS, seed). A fingerprint mismatch fails the suite.
- Trajectory goldens come from the `trajectory` subcommand (direct
  `DALEC_1100()` call) — NEVER from `CARDAMOM_RUN_MODEL.exe` output, which
  writes stale trajectories for prerun-EDC-failing samples (DALEC_MLF2.c:47).

## Debugging a failed equivalence test

1. Assert input bit-identity first (`np.asarray(x).view(np.uint64)`).
2. Run the module with `debug=True` → named intermediates mirroring C locals;
   diff against the tap capture; the first divergent name is the defect site.
3. Classify: transcription bug (fix) / XLA op semantics (substitute the
   bit-matching formulation, document) / irreducible transcendental ULP
   (tolerance override citing a TOLERANCES.md census row).

## Session workflow (agentic)

- Start: read `plan.md` (module flags) + last `CHANGELOG.md` entry.
- End: update both; record failed attempts in CHANGELOG so they are not
  retried blindly. Commit on `jax-port`; never commit to `dalec-baseline`.
- Model/version pins: jax==0.11.1, Python >= 3.11, x64. Upgrading any of
  gcc/glibc/jax invalidates all goldens: regenerate + full rerun.
- **Always run python/pytest with `env -u LD_LIBRARY_PATH`** — the login
  shell's `/usr/local/cuda/lib64` shadows the pip CUDA wheels and silently
  drops jax to CPU (see CHANGELOG 2026-08-23).
