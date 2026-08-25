# dalec_jax

A JAX port of CARDAMOM's **DALEC_1100** (forward model + the 15 live EDCs +
the DALEC_MLF2 likelihood) whose defining property is **verified numerical
equivalence with the C code**. The unmodified C under `../../C/` is compiled
into an oracle (`../../C/projects/JAX_VALIDATION/oracle_1100.c`) that
produces golden outputs, and the test suite asserts the JAX implementation
reproduces them to ≤1e-10 per timestep, per variable.

On top of the verified model sits an **inference fast path**
(`dalec_jax.inference`): exact gradients → multipoint Laplace optimization,
feasible-start screening, and covariance-shaped Metropolis — each measured
against the C's own MCMC and adversarially audited (see below).

> **New to this work? Start with [FINDINGS.md](FINDINGS.md)** — the report
> for the CARDAMOM group: the key questions answered up front, the two
> developments in detail, the C-code findings, and every figure.

## Contents

1. [Verified equivalence](#verified-equivalence-all-gates-green-2026-08-23)
2. [Quick start](#quick-start)
3. [Inference fast path](#inference-fast-path-measured-audited)
4. [Performance, honestly](#performance-honestly)
5. [Environment gotchas](#environment-gotchas-read-before-filing-a-numbers-bug)
6. [Layout](#layout) / [Docs map](#docs-map)

## Verified equivalence (all gates green, 2026-08-23)

| Level | Scope | Result |
| --- | --- | --- |
| L1 | 16 leaf modules × ~4.1k golden cases | 11 modules **bit-identical**; daylength ≤4 ULP (jnp.arccos row); exp/erfc chains ≤1e-12 mixed (census-cited) |
| L4 | 120 full 240-step trajectories | 94 pointwise ≤1e-10 (break step + zero tail exact); 26 chaos-certified — divergence onset inside the C's **own** 1-ULP self-divergence envelope (K=8 parameter dithers through the C oracle) |
| L2/L5 | 15 EDCs × 4000 posterior samples | 60,000 recorded slots, **zero mismatches** (booleans, −inf/NaN sentinels, short-circuit masks exact; operator applied to the C's own trajectories) |
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
  Nothing was silently "fixed" — the two engines stay interchangeable.

## Quick start

```bash
pip install -e ".[test,inference]"
make -C ../../C/projects/JAX_VALIDATION            # build the oracle
python tools/gen_fixtures.py                       # generate goldens (~1 min)
env -u LD_LIBRARY_PATH pytest                      # 30-test equivalence suite
```

The demo driver, the 4,000-sample reference posterior, and a viable
ensemble ship in `tests/data/`, so golden generation and the examples are
self-contained. Then run the fast-path demo:

```bash
env -u LD_LIBRARY_PATH XLA_FLAGS=--xla_disable_hlo_passes=algsimp \
    python examples/laplace_fast_path.py --starts 8 --iters 200
```

Requires Python ≥3.11, jax pinned exact (see pyproject). `jax_enable_x64`
is switched on at import; the equivalence suite is CPU-only by policy and
sets `XLA_FLAGS` itself via `tests/conftest.py`.

## Inference fast path (measured, audited)

`dalec_jax.inference` packages the pieces that exact gradients unlock.
Every number below was measured on the bundled demo site against the C's
own production MCMC, then put through **two rounds of independent
adversarial audit** (fresh-seed replications, re-scoring through the C
oracle, designed counter-experiments); corrections are on the record in
the research archive's audit reports.

```python
from dalec_jax.inference import (build_logpost, find_feasible_starts,
                                 multipoint_laplace, dedupe_modes,
                                 exact_hessians, cap_covariance,
                                 evidence_weights, run_rwm)

logpost, cbf = build_logpost("tests/data/example_1100.cbf.nc")  # z-space, C-identical target
z0, P0, n = find_feasible_starts(logpost, 32)     # ~20 s/start on an A100
res = multipoint_laplace(logpost, z0)             # chunked vmapped L-BFGS
keep = dedupe_modes(res["z_end"], res["P_end"])
covs = [cap_covariance(H) for H in exact_hessians(logpost, res["z_end"][keep])]
out = run_rwm(logpost, z0, covs[0])               # Laplace-shaped Metropolis
```

What each piece is worth (demo site):

- **Mode-finding beats the chain**: 15 of 16 optimizer modes sat at or
  above the best sample the production MCMC ever stored (C-oracle
  re-scored); no chain sample within 2.8 log-units of the best mode. The
  MAP-minus-chain-best gap is a free per-site **convergence audit**.
- **Feasible starts become a bounded step — with one sharp caveat**: the
  full-gate prior pass rate is ≈6e-7 here and ≈5e-6 across eight converted
  FluxVal drivers, so blind batched rejection finds a start in roughly
  1–3M evaluations (tens of seconds on one A100) and has no annealing
  schedule to get stuck in (the single-chain MCMCID-119 search never
  terminated at this site in two attempts). **But blind rejection cannot
  tell "rare" from "impossible."** Two FluxVal sites returned zero hits in
  6e7 draws each and were nearly reported as physically hard sites; their
  feasible sets were in fact *empty*, because a driver bug zeroed snowfall
  and the trajectory EDC divides by that pool's total input (see
  `BUG_COMPAT.md`). Always pair a null screening result with a check that
  the target is satisfiable at all.
- **Error bars in minutes**: after capping likelihood-flat Hessian
  directions at the prior's width (`cap_covariance`), the Laplace
  mixture's per-parameter spreads match the MCMC to median ratio **0.92**
  (was 2.22 uncapped), means to 0.035 in u-space. Screening quality — a
  symmetric bell cannot represent this posterior's skew; the MCMC remains
  the referee for publication numbers.
- **Best Metropolis proposal shape tested**: the capped-Laplace covariance
  gives 2.4–3× the mixing increment of per-parameter tuning and matches
  the "oracle" full chain covariance on median-parameter mixing. But no
  shape rescues plain RWM here (≥~2e4 iterations per effective sample) —
  see the diagnostics caution in `inference/rwm.py`.
- **Known dead ends (measured, don't repeat them)**: vanilla importance
  reweighting of the Laplace mixture collapses in 89-D (weight ESS 1 of
  65,536); NUTS freezes on the hard-EDC target (step size → 2.5e-13) and
  loses ~6× ESS/s to Metropolis even on a cliff-free variant; extra
  optimizer starts beyond ~32 buy nothing (256 starts: +0.36 log-units,
  same basin).

## Performance, honestly

Measured on 2× EPYC 7H12 (256 cores) + 2× A100-40GB; C at `-O2`; full
pipeline (model + EDCs + likelihood) per sample:

| Configuration | ms/sample |
| --- | ---: |
| C `-O2`, 1 core | 0.437 |
| C `-O2`, 256 processes | 0.0068 |
| JAX, 1 pinned CPU core | 0.692 |
| JAX, 1×A100 (batch 64k) | 0.0077 |
| JAX, 2×A100 (batch 64k) | **0.0063** |

- Forward runs: per core the C wins 1.6×; per box, two A100s ≈ the whole
  256-core node. **Nobody should port to JAX for forward throughput.**
- Gradients (the point of the port): exact reverse-mode gradient of the
  log-posterior over all 89 parameters in 0.083 ms/sample at batch on one
  A100 vs 83.5 ms for C central finite differences on one core (~1,000×;
  ~16× the whole node). FD agrees with autodiff to ~1e-14 of the gradient
  scale — i.e. down to the FD's own roundoff floor (~1e-8 per component).
  After value-identical NaN-hardening, gradients are finite for 3,613 of
  3,614 posterior samples.
- Sampling verdict: the C random-walk Metropolis is the most
  cost-effective *sampler* per core on this single-site problem — a
  measured result. The port's wins are mode-finding, curvature, batch
  throughput, and the diagnosis of *why* gradient samplers fail here
  (hard-EDC cliffs; soft-EDC formulations are the untested prerequisite).

All benchmark scripts, raw run artifacts, and the two adversarial audit
reports live in the companion research archive (contact below).

## Environment gotchas (read before filing a numbers bug)

1. `XLA_FLAGS=--xla_disable_hlo_passes=algsimp` for any equivalence-grade
   run (the test suite sets it itself).
2. float64 is enabled at package import and asserted; don't fight it.
3. If JAX cannot find CUDA on a machine whose login shell exports
   `LD_LIBRARY_PATH=/usr/local/cuda/...`: run with `env -u LD_LIBRARY_PATH`
   — the pip CUDA wheels must not be shadowed by a system CUDA.
4. jax is pinned exact; bumping it requires full golden regeneration and a
   suite rerun (`CLAUDE.md`, toolchain-drift policy).
5. Never use `CARDAMOM_RUN_MODEL.exe` output as a numerical reference: on
   prerun-EDC-failing samples it writes the *previous* sample's trajectory
   (see `BUG_COMPAT.md`). The oracle calls `DALEC_1100()` directly with
   per-sample zeroed buffers.

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
  inference/       fast path: target, screening, multipoint Laplace + the
                   prior-width covariance repair, covariance-shaped RWM
examples/          laplace_fast_path.py — end-to-end demo on bundled data
tools/             index generator, fixture/golden generator, ULP census,
                   benchmark + gradient scan
tests/             oracle-driven equivalence suite (TOLERANCES.md = contract)
tests/data/        demo CBF + 4,000-sample reference posterior + viable
                   ensemble (makes goldens & examples self-contained)
```

## Docs map

- `BUG_COMPAT.md` — catalog of C defects reproduced bit-for-bit (each with
  file:line), independently line-verified.
- `tests/TOLERANCES.md` — the tolerance contract: ULP census, chaos
  certification, every override cites a measured cause.
- `CHANGELOG.md` — session log; `plan.md` — per-phase gates; `CLAUDE.md` —
  transcription rules for agentic development; `PR_DRAFT.md` — prepared
  upstream PR text.
- Methodology follows arXiv:2606.07681 (oracle-first LLM translation of
  legacy scientific code), with the tolerance bar tightened ~8 orders of
  magnitude.

Maintainer: Sudhanshu Pandey (pandeysu@caltech.edu). Scope caveats: all
sampler/search/Laplace numbers are from the bundled single demo site at
monthly resolution; likelihood filter modes 1, 2, 4–9 are implemented but
not yet golden-covered (the demo CBF exercises modes 0 and 3).
