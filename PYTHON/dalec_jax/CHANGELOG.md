# dalec_jax session log

Newest first. Every agentic session appends: what was attempted, what passed,
what FAILED (so it is not blindly retried), environment changes.

## 2026-08-29 — Chaos-verdict library: batch-invariant, magnitude-aware

**Library infrastructure only.** This lands `src/dalec_jax/verification.py`,
the one tracked copy of the L4 verdict, plus its tests. **No tracked runtime
path consumes it yet.** `runs/*/verify_against_c.py` is untracked, still builds
its dithers from one batch-indexed stream, and therefore still carries the
defect below; the reported bug is NOT fixed in the research pipeline until that
migration lands (plan.md P9). Do not read this entry as "the bug is fixed".

Triggered by `runs/20260829_BRSa1_mechanistic_ensemble/bug_report_vector46/`:
`laplace_modes[1]` was reported a *genuine* C/JAX discrepancy on two forcing
paths in a 56-vector batch and *chaos-certified* on one of them when run alone.
Reproduced exactly. **No model change** — the transcription is faithful; the
defects are in the verdict harness, and the bug report's own diagnosis was
wrong.

- What actually diverges is **flux 87 `nonleaf_mortality_factor`, C 0.0 vs
  JAX 1.0** on both paths — not "2 ULP on a pool at 1.4e9". The stand dies
  (live carbon 375 → 0.51 gC m⁻² over three timesteps, then 6.7e-17 three
  steps after that); the C's `C_lab/C_roo/C_woo` arrive at exactly `+0.0` by
  **cancellation in an absorbing pool update** (`0 − 0·AMF = 0` thereafter)
  while JAX's land on small non-zero values;
  `ALLOC_AND_AUTO_RESP_FLUXES.c:65` guards on
  `POTENTIAL_AUTO_RESP_MAINTENANCE == 0`, so C takes `NMF = 0` and JAX takes
  `1/exp(≈0) = 1`. Verified by feeding both states through the module directly.
  **No underflow or denormal is involved**: the C steps `−8.30e-26 → +0.0`
  directly (gradual underflow would need to cross 2.2e-308) and every JAX
  value in the chain (−1.52e-43, −4.63e-44, −1.55e-62) is a normal float64.
  Which of the four pool-update passes produces the exact zero was not
  isolated and is not asserted here.
  Pool 11 is **E_LY1, soil thermal energy in J m⁻²**, not carbon — 1.4e9 is
  an ordinary magnitude for it (`UT_TEMP_2_ENERGY.c`: vhc 1.3e6 J m⁻³K⁻¹ ×
  depth × T), and on the 2004-07 path it is bit-identical at the onset step.
  `one_over_deltat = 1/deltat` in both engines.
- NEW `src/dalec_jax/verification.py`. Element criterion **unchanged**
  (1e-10 mixed / 1e-12 absolute) and the certification rule **unchanged** (JAX
  onset ≥ C self-onset − CHAOS_MARGIN); no second route to a certificate is
  added. What changes is how the rule's inputs are obtained.
  1. `dither_block()` seeds from a blake2b hash of the vector's own bits. The
     old `rng.random((n, K, nopars))` indexed one stream by batch position:
     measured C self-onset **128 at position 0 and −1 at positions 1, 2, 3, 5,
     10** on `D_noevent_neutral__donor_2006-07` — the reported flip.
  2. `adjudicate_block()` re-evaluates **every** sample at batch width 1, not
     just the ones that looked dirty in the block. `jit(vmap(...))` is not
     bit-identical across widths in either direction (measured: 7 of 17
     samples move on the 2004-07 path), so canonicalising only the dirty ones
     would record a block-clean/single-dirty sample as CLEAN — the same
     position-dependence, mirrored. Both directions are regression-tested.
  3. `adjudicate()` may escalate K=8 → 64 before reporting a genuine
     discrepancy. Escalation is **monotone and one-sided** (`c_self` is a min
     over K onsets, so more dithers can only move DISCREPANT →
     CHAOS_CERTIFIED) with **no false-positive-rate control**, and it is **not
     what repairs either motivating path**: 2006-07 already certifies at K=8
     and 2004-07 never certifies at all.
  4. `DivergentElement` names the element that failed with |Δ|, the scale the
     criterion divided by, the relative excess and the ULP distance, NaN
     ranked last. The old report named a drifting pool instead, which is what
     sent the bug report down a tolerance-shaped dead end. `as_dict()` is
     strictly JSON-safe (non-finite floats → `None`).
  5. `state_plausibility()` is a **separate axis**: live carbon below
     `LIVE_CARBON_FLOOR = 1e-6` gC m⁻² is recorded as "numerically dead", and
     never softens the agreement verdict. Both paths sit at 6.7e-17 /
     8.7e-18 gC m⁻² at the onset.
- Deterministic dither counts with the shipped seed (20260828), measured to
  K=512: `donor_2004-07` **0 at every K through 512**, so the C really is
  1-ULP-insensitive there; `donor_2006-07` **2/8, 3/16, 3/32, 6/64, 7/128,
  15/256, 28/512** (5.47%), `c_self = 128` at every K.
- Verdicts after the fix, identical alone and at positions 0/27/55 of a
  56-vector block on both paths: `donor_2006-07` **chaos-certified**
  (c_self 128, 2/8 dithers); `donor_2004-07` **discrepant** and **flagged
  implausible**, so it is rejected for the reason that is true rather than
  filed as an implementation defect.
- Ablation over the target at positions 0..3 of a 4-vector block
  (`agreement` per position; the shipped harness is the first row):

  | seeding | escalation | 2004-07 | 2006-07 |
  | --- | --- | --- | --- |
  | position-indexed | none | discr ×5, invariant | **chaos, chaos, discr, discr, discr — NOT invariant** |
  | per-vector | none | discr ×5, invariant | chaos ×5, invariant |
  | position-indexed | K→64 | discr ×5, invariant | chaos ×5, invariant |
  | per-vector | K→64 | discr ×5, invariant | chaos ×5, invariant |

  Per-vector seeding is what makes the verdict well-defined; escalation
  repairs this particular vector too, but only by accident of it being
  1-ULP-sensitive at all.
- **What certification still does not cover**, now documented in
  `tests/TOLERANCES.md` and the module docstring: the verdict is keyed on the
  FIRST onset, so an unrelated error planted after a certified onset is
  invisible (three late-step mutations on the 2006-07 path give a
  byte-identical verdict). Also recorded there: a dither whose trajectory goes
  non-finite is a genuine self-divergence under the element criterion and can
  certify — the rule applied as written, unchanged from `gen_fixtures.py`.
- NEW `tests/test_verification.py` (21 tests). Oracle-gated set asserts the
  same verdict alone and at positions 0/27/55 of a **56-vector** block — the
  original width — on both donor paths, with 55 healthy companions (rows of
  the tracked `assim_1100.cbr` that run to completion on both paths). The
  oracle fixture now FAILS on a non-zero oracle exit and skips only when the
  binary is absent or its shared libraries are unresolvable before execution.
  The dither test asserts the legacy construction *did* move with position, so
  it is not vacuous. `tests/data/chaos_D_donor_{2004,2006}-07.cbf.nc` +
  `chaos_laplace_modes_1.par.txt` committed (110 KB).
- `tests/test_trajectory.py` and `tools/gen_fixtures.py` now import the shared
  criterion (three copies had drifted apart). Value-identical over 360
  adversarial fixture pairs: no golden impact. `gen_fixtures.py` keeps its
  position-indexed dither draw on purpose — the fixture set is regenerated as
  a unit — with a comment saying why and that switching it needs `make golden`.
- NOT in this change, filed separately: `feasible_starts[12]` and `[27]`
  differ from the C on the **EDC feasibility decision** on all 10 paths. That
  is a pass/fail disagreement, not a trajectory difference — different
  signature, different fix. See FINDINGS.md §4.4.

## 2026-08-24 — Inference fast path + self-containment + README overhaul

- NEW `src/dalec_jax/inference/`: `target.py` (z-space log-posterior,
  C-identical target, with the Jacobian-datum caution), `screening.py`
  (batched prior rejection; measured pass rate ~6e-7 at the demo site,
  audit-replicated and C-oracle-confirmed), `laplace.py` (chunked vmapped
  L-BFGS — monolithic 400-iter scan costs ~45 min of XLA compile, chunks
  of 20 compile in seconds; exact Hessians; `cap_covariance` = the
  prior-width repair of flat directions, spread ratio 2.22->0.92 median;
  evidence weights; explicit warning against 89-D importance reweighting,
  measured ESS 1/65,536), `rwm.py` (vmapped Metropolis with arbitrary
  proposal covariance; capped-Laplace shape measured best of four, 2.4-3x
  diagonal, with the multi-chain-ESS-is-diversity caution from the
  same-start control audit). Optional dep: `pip install ".[inference]"`.
- NEW `examples/laplace_fast_path.py`: end-to-end demo (starts -> modes ->
  Hessians -> capped mixture) on bundled data; `--screen` for blind
  rejection starts.
- Self-containment: `tests/data/{example_1100.cbf.nc, assim_1100.cbr,
  viable_ensemble.cbr.nc}` committed (~3 MB); `tools/gen_fixtures.py` now
  reads them from `tests/data/` instead of a sibling research checkout —
  `make golden` + pytest + examples run from a bare clone + C toolchain.
- README overhauled: equivalence table kept; added audited performance
  tables, the inference fast path with its measured worth and known dead
  ends, environment gotchas (incl. the LD_LIBRARY_PATH CUDA shadowing
  fix and the CARDAMOM_RUN_MODEL stale-output warning), docs map.
- All sampler/Laplace numbers cited in code/docs went through two
  independent adversarial audits (reports in the research archive);
  corrected statements are the ones quoted here.

## 2026-08-23 — Session 1 (P5+P6, same session)

- P5: likelihood layer ported + L6/L7 green (30 passed). Three new
  BUG_COMPATs (obsope_int_div_index, obsope_c3frac_unset, unused_unc_9999).
  L7 three-clause gate over all 4000 posterior samples; K=4 dithers were
  insufficient for 2 posterior-relevant holdouts (2523: JAX dP=-3.67 vs
  K=16 C-dither spread [-inf,+0.19] incl -10; 3087 similar) -> posterior
  dither battery is K=8 with a gate-instability clause. PEQ_CUE 0/0-ratio
  chaos (MGPP->0) produces astronomically-rejected P on both sides
  (both-rejected clause).
- P6: benchmarks + grad scan (see plan.md). A100 full-pipeline vmap(4000)
  = 0.029 ms/traj (6.7x one C core); CPU parity-class. grad(P) finite for
  47/64; offender sites named; hardening deferred with value-identity
  requirement.
- CBF stream configs actually exercised by goldens: GPP(f0,u2+unc series),
  LAI/ABGB(f3,u1), EWT(f0,u0,norm1), ROFF(f0,u1), SCF(f0,u0,thresh),
  Mean_FIR/PEQ_CUE/PEQ_iniSOM singles. Filter modes 1,2,4-9 implemented
  but NOT golden-covered — synthetic-CBF coverage is future work.

## 2026-08-23 — Session 1 (P3, same session)

- Full forward model ported (model/dalec_1100.py): VegK prederive in numpy
  (bit-identical libm), init block, literal step transcription with the
  4-pass carbon removals, post-overflow q_ly1/q_ly3 ordering into the
  energy fluxes, and the q_ly1_overflow bug reproduced. lax.scan carries
  (pools, alive); the isfinite break emits the poisoned step then zeros.
- **L4 green: 22 passed.** 94/120 fixtures pointwise (mixed ≤1e-10 or abs
  ≤1e-12; break step + zero tail exact), 26 chaos-certified.
- **Findings (do not re-derive):**
  - DALEC_1100 is ULP-chaotic for ~29% of fixture draws: the C diverges
    from ITSELF under 1-ULP parameter dithers (all-89-param, K=8) with
    onsets bracketing the JAX onsets — JAX-vs-C is indistinguishable from
    C's own ULP sensitivity. Certification lives in chaos_cert.json and
    MUST use the identical pools+fluxes criterion as the test (pools-only
    onsets misclassify flux-flip fixtures 10/73/111).
  - The early "failures" at t=0-2 were hydraulic_mortality_factor
    cancellation noise (|Δ| ~1e-16 on values ~1e-15) — hence the 1e-12
    absolute escape in the element criterion.
  - The HMF exact-equality gate (LF1+LF2)==2 never flipped across all 120
    fixtures — the chaos seeds are the smooth erfc/exp ULP + the
    dlambda_dt sign discretization, not the equality gate.
  - CBF time variable is named 'time' (READ_NETCDF maps it to TIME_INDEX).
- Perf note (unoptimized): CPU vmap(120) ≈ 15 ms/trajectory vs C 0.196 ms —
  scan dispatch dominates on CPU; do not quote before P6 GPU/batching work.

## 2026-08-23 — Session 1 (P2, same session)

- All 16 leaf modules ported (constants.py + modules/*) and L1 green:
  11 bit-identical, COMPUTE_DAYLIGHT_HOURS ≤4 ULP, four exp/erfc chains
  within mixed-criterion bounds (worst: LIU transp 2.6e-13 — exp ULP
  through (co2−ci) conditioning; bound 1e-12).
- **Repair-loop findings (do not re-derive):**
  - jit ≠ eager numerically: XLA's `algsimp` pass rewrites div-by-const to
    mul-by-reciprocal and folds (x/c1)·c2 → x·(c2/c1) — confirmed in
    optimized HLO of the INITIALIZE_INTERNAL_SOIL_ENERGY kernel.
    `--xla_disable_hlo_passes=algsimp` restores bit-identity and is now
    enforced by tests/conftest.py (set BEFORE jax import). None of
    --xla_allow_excess_precision=false / fast_math / optimization_level=0 /
    disable fusion helped — it is algsimp specifically.
  - jnp.arccos: 1 ULP off glibc on ~3% of args (matches census) → DAYL ≤4.
  - Raw-ULP comparison is meaningless for denormal-adjacent outputs
    (mortality factors ~1e-290): use the mixed |Δ|/max(|ref|, col-RMS)
    criterion; kept exact-class where census says bit-identical.
  - jnp.fmax/jnp.fmin (NOT maximum/minimum) reproduce C fmax/fmin NaN
    semantics.
- pytest: 20 passed, 1 skipped (GPU smoke skips under JAX_PLATFORMS=cpu).

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
