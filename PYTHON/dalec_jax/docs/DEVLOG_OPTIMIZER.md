# Devlog: is the "MAP" actually the MAP? L-BFGS vs true Newton

A running record of an open investigation. Numbers on this page are
measured; anything not yet measured is marked TBD.

Status (2026-08-31): the original question is answered, and the answer
changed twice. Reading order matters — the four-arm comparison concluded
that "the MAP is ill-defined at NL-Loo", and the production-ADEMCMC control
run at the bottom of the page **overturns that conclusion**. Two other
claims made along the way were refuted by their own verification (gate-free
curvature; a buffered-output timing artifact). Corrections are kept in place
rather than edited away, because the sequence is the point.

Contents: the defect → toy pre-tests → ridge atlas → H100 inversion →
refuted hypotheses → four-arm comparison → **the NaN-Hessian fix** →
**SARLA** → **the ADEMCMC control that revised the story**.

## The defect that started this (2026-08-28)

A reader asked why, in a posterior figure for NL-Loo, the MAP star sat far
from the chain's densest region. Scoring the stored "MAP" mode and all
6,400 stored RWM draws with the same z-space log-posterior:

| point | z-space log-posterior |
| --- | --- |
| stored "MAP" mode (200-iter L-BFGS, pilot) | **−524.7** |
| best stored chain draw | **−233.7** |
| chain draws beating the "MAP" | **5,412 / 6,400 (85%)** |

So at NL-Loo the pilot's mode is not the MAP — the 200-iteration L-BFGS
stalled partway up the site's litter trade-off ridge (`t_lit` × `i_lit`,
correlation −0.97, visibly curved), and the chains it seeded walked past it.

Two lessons already bankable:

1. **The demo-site result "15/16 modes ≥ chain best" does not transfer to
   every site.** The MAP-minus-chain-best gap must be computed per site —
   which is cheap, and is itself the convergence audit we recommend.
2. A limited-memory quasi-Newton method is structurally weak on bent,
   highly-correlated ridges: its low-rank curvature history goes stale as
   the ridge turns.

## The hypothesis

L-BFGS uses only gradients (`optax.lbfgs` + `jax.value_and_grad`); the
exact JAX Hessian is currently computed **only after** optimization, for
the Laplace covariance. If curvature staleness is the bottleneck, a damped
Newton method using the exact Hessian at **every step** should traverse
the curved ridge in far fewer iterations — at ~89× the per-iteration cost
(one forward-over-reverse Hessian vs one gradient).

## The test (scripts/newton_vs_lbfgs.py in the research repo)

One site (NL-Loo, the failure case), identical starts (the site's 16
stored screening hits), all endpoints re-scored by the unmodified C oracle:

| arm | method | budget |
| --- | --- | --- |
| A | L-BFGS (pilot baseline) | 200 gradient iters |
| B | L-BFGS extended | 1,000 gradient iters |
| C | Damped Newton (Levenberg), exact vmapped 89×89 Hessian per iter | 60 Newton iters |
| D | Hybrid: A's endpoints + Newton polish | 200 g + 15 H |

Reference bar: the best stored chain draw (−233.7). An arm is only
interesting if it reaches or beats the chain.

Newton damping: `dz = −(H + λI)⁻¹ g`, per-start adaptive λ (accept →
λ/3, reject → λ×10) — large λ degrades toward a short gradient step, so
starts adjacent to hard-EDC cliffs don't diverge.

## Results

### Toy pre-test (2026-08-28): the valley must be walked either way

Before spending GPU time on the real target: 50-D generalized Rosenbrock
(a long, bent, tightly-correlated valley — the NL-Loo ridge in caricature),
8 shared random starts, float64, CPU. Newton = exact-Hessian damped
(Levenberg with PD safeguard + backtracking line search).

| method | iterations to f < 1e-6 (median of 8) | cost per iteration |
| --- | --- | --- |
| L-BFGS (optax, zoom line search) | **~265** | 1 gradient (+ line search) |
| exact-Hessian Newton + line search | **~125** | 1 Hessian ≈ D gradients + eigsolve |
| naive damped Newton (no line search, no PD guard) | stuck at f ≈ 27 after 60 | — |

Three findings:

1. **Curvature only halves the step count; it does not shortcut the
   valley.** Both methods must traverse the same arc; Newton's quadratic
   model is only locally valid on a bending ridge, so its precise steps
   are barely longer than L-BFGS's cheap ones. At DALEC cost ratios
   (one 89-D Hessian ≈ 89 gradients) Newton's traversal costs ~10–40×
   more compute than L-BFGS's.
2. **The pilot's failure mode reproduces as a budget problem**: L-BFGS at
   200 iterations sits at f ≈ 9 (far from converged); at ~265 it drops to
   machine precision. Prediction for the real target: arm B (L-BFGS 1000)
   fixes NL-Loo; arm C (Newton) is not wall-competitive for traversal.
3. **Naive Levenberg damping without a line search is genuinely bad**
   (rejected steps discard whole Hessian evaluations) — worth knowing
   since that is the cheapest Newton one would write first.

Where Newton should still earn a place: the **endgame**. Its quadratic
convergence (f: 8.9 → 0 between iterations 100 and 200, i.e. machine-zero
once inside the basin) certifies stationarity exactly where the Laplace
Hessian is about to be computed — a ~10-iteration polish after L-BFGS,
not a replacement (arm D).

### Smoke on the real target (4 starts, 15 Newton iters): three lessons, one bug

Measured 2026-08-28, one A100, NL-Loo, 4 weak screening hits as starts:

| arm | best P (z-space) | wall |
| --- | --- | --- |
| L-BFGS 100 iters | −1,937.9 | 1,654 s (16.5 s/iter) |
| naive damped Newton, 15 iters | −5,296.0 (stalled, λ → 3×10⁶) | 729 s (~48 s/iter after 342 s compile) |
| Newton polish of L-BFGS endpoints, 5 iters | no improvement | +13 s |

1. **The toy's warning reproduced exactly on the real target**: full-step
   Levenberg damping stalls — λ escalated six orders of magnitude with
   nearly every step rejected. The full run now uses the toy-validated
   backtracking line search along the Newton direction.
2. **The per-iteration cost surprise (GPU):** a Newton iteration cost only
   **~3×** an L-BFGS iteration (48 vs 16.5 s at batch 4) — nowhere near
   the naive 89×. The Hessian's 89 forward-over-reverse columns are
   parallel work that actually fills the latency-bound GPU, while the
   gradient leaves it idle. On GPU, Newton's handicap is small; with
   ~2× fewer iterations needed (toy), it may even be wall-competitive.
3. A `z2par` type bug crashed the final C-re-score stage (tuple vs
   ndarray) — fixed; optimization results above were unaffected.

### Side result: combining different-point Hessians (the "ridge atlas")

Toy (2-D banana, ground truth by quadrature): a single-mode Laplace has
KL(truth‖approx) = 11.1 and misses the arms entirely (y-spread 0.50 vs
truth 1.92). A mixture of local Gaussians at points along the ridge —
each Hessian's Gaussian centered at the **once**-Newton-shifted point
z − H⁻¹g, covariances capped as usual, weighted by ridge height — gives
**KL 1.39 with three Hessians and 0.55 with five** (a 20× improvement).
Two hard-won details: the shift must be applied once, not iterated
(iterating collapses every piece to the global mode), and an indefinite
off-crest Hessian must be variance-capped, not PSD-floored (the floor
turns it into a giant flat blob that swallows the mixture weight).
Candidate upgrade for the screening pipeline at bent-ridge sites like
NL-Loo; needs a real-target test against the RWM ensemble.

**And the exchange rate vs Laplace-guided RWM, same toy** (64 chains,
capped-Laplace proposal, histogram density, KL(truth‖approx)):

| approximation | model evaluations | KL |
| --- | --- | --- |
| single capped Laplace | ~100 (mode find + 2 Hessians) | 11.12 |
| ridge atlas, 5 pieces | ~1–2k grad-equivalents (10 Hessians) | **0.55** |
| RWM, 64×200 | 12,800 | 44.3 (worse than Laplace — chains haven't spread) |
| RWM, 64×1,000 | 64,000 | 9.48 (≈ single Laplace) |
| RWM, 64×5,000 | 320,000 | 0.031 |
| RWM, 64×20,000 | 1,280,000 | 0.008 |

Reading: the chain needs roughly **100–150k evaluations to match what the
atlas gets from ~10 Hessians** (a ~50–100× compute advantage at screening
budgets), and early-budget MCMC is *worse than a single Laplace* because
unspread chains leave the arms empty. But only the chain keeps improving:
by 320k evals it is 20× better than the atlas can ever be — straight
ellipses tile a curve only so well. Division of labor confirmed: atlas
for the screening answer, MCMC for the publication answer.

### Ridge atlas on the real target (BE-Vie, 89-D, 2026-08-28)

Protocol: referee = the site's stored 64-chain RWM ensemble; anchors = 8
high-P chain draws spread by farthest-point sampling; one Newton shift +
capped Hessian per piece; weights ∝ ridge height. 12 min on CPU
(~8 s/Hessian warm; the same Hessian is 0.24 s on the H100).

| metric (89 parameter marginals vs chain) | single Laplace | atlas (7 pieces) |
| --- | --- | --- |
| median KL | 0.406 | **0.238** |
| worst-parameter KL | 10.72 | **1.02** |
| params where atlas is better | — | 67/89 |
| median width ratio (approx/chain) | 1.27 | 1.54 |

Reading: on an easy, mostly-straight-ridge site the median gain is modest
(1.7×), but the **worst-case marginal improves 10×** — the atlas earns its
keep exactly on the bent/boundary parameters that break a single Gaussian.
The honest wrinkle: the atlas is **over-dispersed** (widths 1.54× the
chain vs 1.27× for single Laplace) — overlapping prior-capped pieces add
spread-of-means variance. Fine for screening (conservative), wrong to
quote as posterior widths without repair.

Field notes: the single best chain draw had a **non-finite Hessian**
(sits against an EDC cliff — more evidence the chain concentrates where
curvature is undefined), and the Newton shifts moved chain-draw anchors
essentially nowhere (they are already on the crest; the shift matters for
off-support anchors like optimizer iterates).

### Ridge atlas on the hard site (NL-Loo, 89-D): an instructive failure

Same protocol, K=8, run on an idle H100 NVL (3.2 min wall vs 12.2 min on
CPU; warm 89×89 Hessian 0.24 s vs ~4 s on A100 — the H100's FP64 is
~17× faster per Hessian). Result: the atlas **degenerated to the single
Laplace** (median KL 0.577 vs 0.578; weights [1, 0, 0, 0]). Two causes,
neither visible in the 2-D toy:

1. **Cliff Hessians are pervasive on the hard site.** 4 of 8 high-P chain
   draws had non-finite Hessians — they sit against hard-EDC boundaries
   where curvature is undefined (BE-Vie lost only 1 of 8). The chain
   concentrates exactly where the quadratic model does not exist.
2. **Ridge-height weighting is wrong in high dimensions.** The usable
   pieces spanned ~40 log-units of P along the ridge, so exp(P) weights
   annihilated all but the top piece. In 89-D the typical set lives far
   below the density peak: pieces low on the ridge still cover real
   probability mass. Peak height is not mass — the 2-D toy (ridge drop
   ~2 log-units) could not expose this.

Fixes tested (K=24 anchors → 14 usable pieces after cliff casualties;
4.5 min per run on the H100):

| NL-Loo, 89 marginals vs chain | median KL | worst KL | atlas better | width ratio |
| --- | --- | --- | --- | --- |
| single Laplace | 0.57 | 22.0 | — | 1.12 |
| atlas, exp(P) weights (K=8) | 0.58 | 22.3 | 44/89 | 1.13 |
| atlas, evidence weights + floor | 0.43 | 11.7 | 76/89 | 1.17 |
| **atlas, uniform weights** | **0.29** | **4.4** | 70/89 | 1.59 |

**Uniform weighting wins** — median KL 2× better than the single Laplace
and the worst marginal 5× better. Every density-derived weighting
(exp(P), evidence P + ½logdet) collapses onto the top piece, because the
ridge's ~40-log-unit height drop dwarfs any logdet correction; in 89-D
what matters is *covering the typical set*, i.e. tiling the ridge by
arc length, not by height. Even the evidence-weight run's improvement
came entirely from its 0.25% floor pieces — sub-percent mass in the
right place repairs KL's fatal "chain mass where approx has none" error.

The price is over-dispersion (widths 1.59× the chain): uniform tiling
buys shape at the cost of spread. Screening verdict: use uniform-weight
atlas for shape/coverage questions; use the single capped Laplace (1.12×)
when only per-parameter widths are needed; the chain remains the referee
for both.

The methodological point stands regardless of outcome: the
toy → easy-site → hard-site ladder has broken a different hidden
assumption at every rung (iterated shifts collapse the atlas; PSD floors
create weight-eating blobs; exp(P) weighting fails in high-D; cliff
Hessians thin the anchors). None of these failures were visible at the
previous rung.

### H100 changes the Newton verdict (microbenchmark, 2026-08-28)

Warm steady-state per-iteration cost on the real NL-Loo target
(`opt_iter_bench.py`; compiles excluded; L-BFGS = optax lbfgs+zoom,
Newton = exact Hessian + eigendecomposition + 8-point line search):

| device | batch | L-BFGS s/iter | Newton s/iter | ratio N/L |
| --- | --- | --- | --- | --- |
| A100 (smoke, no line search) | 4 | 16.5 | ~48 | ~3 |
| **H100 NVL** | 4 | 5.66 | **0.48** | **0.085** |
| **H100 NVL** | 16 | 6.44 | **0.58** | **0.090** |

On the H100 a full exact-Hessian Newton iteration is **10× cheaper than
one L-BFGS iteration**. Mechanism: the L-BFGS zoom line search issues
many *sequential* latency-bound model evaluations per iteration, while
Newton's cost concentrates in the Hessian — 89 parallel columns that the
H100's FP64 units (0.24 s/Hessian, 17× the A100) chew through, plus a
line search whose 8 candidates batch into one launch. Combined with
needing ~2× fewer iterations (toy), Newton on H100-class hardware is
~20× faster per unit progress — the traversal recommendation *inverts*
between GPU generations. (A100 row to be re-measured with the identical
microbenchmark for apples-to-apples once the full run frees the card.)

### Gate-free curvature: REFUTED by its own verification (branch `edc-cliff-handling`)

Hypothesis: the EDC gate is -inf-or-zero, so differentiating the ungated
likelihood gives exact curvature at feasible points, curing the NaN
Hessians. Implemented as `build_logpost(..., gate="none")`; verified by
`cliff_hessian_test.py`. **Both claims failed:**

1. Identity check: max |hard − none| = **inf** on feasible chain draws,
   and the finite Hessians disagree at rel ~1. The EDCs are NOT pure
   indicators here — they add **continuous penalty terms** (the
   `edc_eqf` equilibrium factors) inside the feasible set. Dropping the
   EDC stage drops real curvature.
2. **0 of 9 NaN Hessians were cured** — the non-finiteness originates in
   the likelihood/model derivative path itself, not the gate. Leading
   suspect: the catalogued missing LAI `min_threshold` (GPP/ET floor
   their operands at 0.1 before the log; LAI does not, and
   dying-vegetation trajectories reach denormal LAI whose log-derivatives
   explode). **Probe result: this hypothesis is refuted too.** Minimum
   D_LAI at the 9 NaN anchors is 0.4–1.4 (healthy canopy, no denormals),
   statistically indistinguishable from the 15 finite anchors; foliar C
   and minimum pool values also don't separate the groups. The NaN cause
   remains open ~~. Next suspect: the known JAX higher-order-derivative
   leak~~ — **localized (debug_nans + HVP direction scan + checkify):**
   the value AND gradient are finite; the NaN is born at second order,
   inside the model `lax.scan`, in a `mul` primitive (the classic
   inf·0 tangent that a double-`where` cannot mask at second order).
   The 12/89 NaN Hessian-vector directions are one functional cluster —
   `LCMA, i_labile, Vcmax25, plgr, k_leaf, lambda_max, phi_RL,
   canopyRdsf, t_lab, rauto_mr_w, i_wood, T_phi` — the **KNORR
   phenology / labile-growth path**, which contains the model's guarded
   `pow`/`exp`/`erfc` sites. Fix shape: a tangent-space guard (third
   `where` or custom_jvp) on the one offending site, which leaves the
   primal bit-exact and so does not violate the equivalence contract;
   to land on branch `edc-cliff-handling` after module-level bisection
   names the exact line, with a before/after test at all 9 bad anchors
   plus the L1 module regression.

Consequences: `gate="none"` is mislabeled as-is — the correct curvature
mode must keep the smooth EDC penalties and neutralize only the -inf
branches; and the cliff story splits in two (soft-vs-hard gating for
samplers, versus a plain missing threshold for derivatives). The fix
hypothesis was killed by its own pre-registered test — which is the
process working, not failing.

### Full L-BFGS-vs-Newton comparison (NL-Loo, 16 shared screening-hit starts, one A100)

| arm | best P (z-space) | best P (C re-score) | wall | evals |
| --- | --- | --- | --- | --- |
| chain best (reference) | **−233.7** | **−58.3** | — | — |
| A: L-BFGS 200 | −402.2 | −217.8 | 2,039 s | 200 g |
| B: L-BFGS 1000 | −402.2 (identical to A) | −217.8 | 7,851 s | 1000 g |
| C: Newton + line search, 60 iters | −2,928.1 | −2,752.6 | 726 s | 60 H+g |
| D: A + 15-iter Newton polish | −402.2 (no gain) | −217.8 | 2,052 s | 200g+15H |

Three verdicts, one of which kills our own earlier prediction:

1. **"It's an iteration-budget problem" — REFUTED.** Arm B ran 5× the
   budget for 2.2 h and finished bit-identical to arm A. L-BFGS is not
   stalled short of the top; it is converged at stationary points ~184
   C-log-units below the chain's best draw. (The toy predicted budget
   would fix it; the real ridge disagrees.)
2. **Arm C is confounded by the KNORR second-order NaN leak** (see
   above): λ pinned at its 10⁸ cap from iteration ~10 — the Newton
   direction is being poisoned exactly as the NaN-Hessian diagnosis
   predicts. Newton on the real target cannot be judged until the
   tangent-space fix lands; re-run scheduled after that.
3. **Nothing beats the chain — decision-matrix outcome 3.** On NL-Loo
   the optimizer arms all plateau far below what stochastic diffusion
   along the flat, curved, cliff-bounded ridge finds. Where a ridge is
   effectively flat along its length, gradient ascent has no signal to
   follow lengthwise while the chain diffuses freely; "the MAP" is
   numerically ill-defined at this site and the sampler's mass answer is
   the only meaningful one. Practical rule adopted: report
   max(mode, chain-best) as the MAP candidate and flag any site where
   the chain wins — that flag is itself the ridge-flatness diagnostic.

Wall-clock footnote: the entire four-arm experiment (3.5 h) cost ~7× one
pilot site posterior; the H100 microbenchmark suggests the same suite
would run ~10× faster there once the Hessian fix makes arm C meaningful.

### Decision: settled by the production-ADEMCMC control (see below)

Outcome 3 held *among the JAX arms* (nothing beat the chain). But the
control run added later overturns the interpretation we drew from it: the
ridge is not intrinsically pathological — our methods were simply too
cheap. See "The control we should have run first".

## The KNORR NaN Hessians: located, fixed, and not finished (2026-08-30)

The leak that confounded arm C above is **not in KNORR phenology at all**.
It is in the gradient-hardened expression

```python
1 / jnp.exp(jnp.minimum(x, LOG_DBL_MAX))
```

at three sites: the ALLOC growth factor and non-leaf mortality factor
(`modules/alloc_and_auto_resp.py`) and the leaf mortality factor
(`modules/liu_an_et.py`). The 12 flagged "KNORR/labile" parameter
directions were downstream victims, not the cause.

**Mechanism.** For `x` just below the cutoff (measured: `x_gf = 707.7897`
at timestep 86, NL-Loo), `exp(x)` is finite-but-huge. The forward-mode JVP
of `exp` forms `exp(x)·dx`, which overflows to `inf` for any tangent of
ordinary size; the following `div` then yields `inf/inf = NaN`. Reverse
mode never forms that product — its cotangent chain divides by
`exp(x)² = inf`, underflowing benignly to 0. **That asymmetry is exactly
why `jax.grad` stayed finite while every `jacfwd(grad)` and HVP went NaN**,
and why the defect survived so long: nothing in the gradient-based pipeline
could see it.

**The non-obvious part: `custom_jvp` does not fix this inside `lax.scan`.**
Forward-over-reverse differentiates the scan body *after* linearization has
decomposed the custom call, so the raw body is re-differentiated and the
overflow returns (verified with a single-step scan micro-repro: eager HVPs
fine, scan HVPs NaN). The body itself must be safe at every derivative
order. The landed guard is a straight-through idiom whose value path is the
original expression and whose derivative path only ever touches the bounded
`exp(-m)`:

```python
m = jnp.minimum(x, LOG_DBL_MAX)
w_stable, w_exact = jnp.exp(-m), 1 / jnp.exp(m)
sg = jax.lax.stop_gradient
return sg(w_exact) + jnp.where(jnp.isfinite(w_stable),
                               w_stable - sg(w_stable), 0.0)
```

### Verification (81 SARLA chart centers × 12 implicated directions = 972 HVPs, equivalence-grade flags)

| check | before | after |
| --- | --- | --- |
| NaN HVP directions (of 972) | 163 | **12** |
| anchors with ≥1 NaN direction (of 81) | 21 | **1** |
| whole-model primal bit-identical | — | 80 / 81 |
| guard in isolation (400,013-point sweep) | — | **0 differing bits** |
| `pytest tests` | — | 44 passed, 1 skipped |

**Two limits, stated rather than buried** (both recorded in the
`ad_guards.py` docstring and the commit message, `edc-cliff-handling`
`cc52936e`):

1. **Incomplete.** Anchor 79 retains all 12 NaN directions, *unchanged by
   the guard* — at least one further overflow site exists elsewhere in the
   step and is not yet localized.
2. **Not globally bit-preserving.** One anchor's whole-model log-posterior
   shifts by 5.7e-14 (~1–2 ulp; ~10⁵ inside the 1e-10 trajectory bar).
   Since the guard is exact in isolation, this comes from XLA fusing the
   enlarged graph differently, not from the arithmetic. Under default flags
   (algsimp on) 4/77 anchors shift by the same magnitude.

A third observation, incidental but worth recording: **with `algsimp`
disabled all 81 anchors are feasible; with it enabled only 77 are.** Four
anchors sit close enough to an EDC cliff that 1-ulp rounding flips them
between finite and −inf. Any exactness claim on this target must pin the
XLA flags.

---

## SARLA: an audited, rank-adaptive Laplace atlas (2026-08-30)

Full specification and results:
[SARLA design note](https://app.notion.com/p/3cd788b60530817a904ff70d7f6ef58b).
Implementation lives in the research repo (`scripts/sarla*.py`).

The idea: build a Laplace atlas, then use short **frozen-proposal audit
epochs** to find where the atlas is wrong, project each discrepancy onto the
near-optimal set (tangent-preserving normal-space re-optimization), diagnose
*why* it failed (missing extent / bend / separate basin / rank change), and
perform atlas surgery — repeating until two consecutive audits come back
clean, then freezing for exact Metropolis–Hastings.

### Where it wins and fails (6 targets, matched eval budgets)

Compared against single Laplace, a static atlas, **AIMM-lite** (same audit,
but components placed *at* the flagged point — the established adaptive
mixture recipe, and the key ablation), and tuned RWM. JS = Jensen–Shannon
divergence of produced samples vs grid truth.

| target | SARLA (KL / acc / JS) | AIMM-lite | RWM (JS) | verdict |
| --- | --- | --- | --- | --- |
| near-Gaussian | 0.015 / 0.96 / 0.009 | identical | 0.028 | control: freezes at K=1, no wasted surgery |
| curved ridge | 0.78 / 0.26 / 0.017 (K=10) | 1.02 / 0.20 / 0.031 (K=17) | 0.015 | win: better fit per chart |
| 3 modes, 1 seeded | **0.094 / 0.76 / 0.006** | 0.94 / 0.34 / 0.011 | 0.219 | flagship win; RWM never crosses |
| branching valley | 0.96 / 0.33 / 0.022 (K=4) | 1.42 / 0.18 / 0.023 (K=15) | 0.014 | win: diagnoses **rank-change** at the bifurcation |
| Neal's funnel | 0.87 / 0.36 / 0.084 (K=1) | identical | **0.046** | **fail**: rank varies continuously; audit never fires |
| Cauchy tails | 0.85 / 0.26 / 0.040 (K=16) | 0.87 / 0.20 / 0.031 | **0.015** | **fail**: Gaussians cannot tile power-law tails |

Three implementation lessons, each found by a failing test rather than by
reasoning: the defensive component must be **prior-wide**, not atlas-wide
(it has to reach mass the atlas knows nothing about); the normal corrector
must start **at the flagged point's own offset** (starting at the chart
centre collapses every diagnosis back onto the known atlas); and freezing
requires **two consecutive clean audits** (a single clean audit was followed
by two real defects in the ridge test).

### Dimension scaling, and the honest limit

Embedding the banana in nuisance dimensions: at d=8 the nuisance eigenvalues
sit inside the spectral gap, the ridge is silently misclassified as rank-0,
and **no repairs land at all** (K=1, ESS 0.002). Loosening the gap threshold
from 10 to 5 fixes d=8 completely — the rank rule works but is brittle in
the 4–6 ratio range, which is precisely the "stability guarantees" gap the
design note lists as a prerequisite for publication. Even with rank fixed,
independence-MH acceptance decays 0.21 (d=8) → 0.10 (d=16) → 0.06 (d=32).

### On the real 89-D target (NL-Loo)

The atlas machinery worked: 17 seeds → **81 charts in 537 s** on one A100,
diagnoses overwhelmingly *rank-change* (tangent count varies 28–51 along the
ridge — correctly sensed), NaN-Hessian charts degrading gracefully to
prior-width. But **global independence sampling collapsed**: audit ESS
pinned at 0.000, production IMH acceptance **0.002**. No tractable mixture
is globally accurate enough in 89-D for independence proposals.

Spending the same atlas *locally* is what worked. Launching 64 chains from
the audit-repaired chart centres with chart-shaped local steps
(acceptance-tuned; the textbook 2.38/√d collapsed again, as it did in the
pilot) reached **P = −200.7 z-space, C-oracle verified −34.24** — a
33-log-unit improvement on the previous project best, in **~12 min of GPU
time** (537 s atlas + 179 s sampling). Atlas jumps in 89-D are real but
marginal: 0.08% acceptance, 1.3 vs 1.0 regions visited per chain.

**Conclusion:** SARLA's audit-and-surgery loop is a validated *atlas builder
and geometry certifier*; its independence-MH stage is a low-to-mid-
dimensional tool. In high dimension, spend the certified atlas locally —
which is the Laplace-guided RWM we already had. Same object, different
resolution.

---

## The control we should have run first: production ADEMCMC on NL-Loo (2026-08-31)

Every claim above about NL-Loo being pathological rested on comparisons
among *our* methods. So we ran the production C sampler on the same site:
**64 independent ADEMCMC chains** (unique seeds, 500,000 iterations,
1,000 stored samples each), niced, on the 256-core node.

| | value |
| --- | --- |
| feasible-start search | median **65 min** (min 59, max 2.2 h) |
| main MCMC | median **31.6 h** (min 27.3, max 32.7) |
| best P (C oracle, 200 subsampled draws/chain) | **−25.56** (median per chain −28.71, worst −31.67) |
| Gelman–Rubin R̂ | **< 1.1 for all 89 parameters** (max 1.07) |

**This overturns two claims made earlier on this page.**

1. **"The MAP is numerically ill-defined at NL-Loo / the ridge defeats
   samplers" — WRONG, or at least far too strong.** ADEMCMC converges here:
   64 independent chains agree (R̂ < 1.1 on every parameter) and reach a
   consistently better region. *Every single chain individually beat our
   best JAX result.* The ridge is hard for cheap methods, not intrinsically
   pathological. The "report max(mode, chain-best) and flag it" rule still
   stands as a diagnostic, but the flag means "your sampler was too cheap",
   not "no MAP exists".
2. **A reported timing was a measurement artifact.** During the run this
   page's author reported that 58 of 64 chains had not cleared the
   feasible-start search after 32 h. That was **wrong**: C stdout is
   block-buffered, so progress lines only appeared when buffers flushed and
   the grep undercounted. Real median search time was 65 min — *faster* than
   the demo site's 2.2 h. Recorded here because the false version was the
   more interesting story, which is exactly when a number deserves a second
   look.

### Where that leaves the speed comparison

| method | wall clock (NL-Loo) | best P (C) | product |
| --- | --- | --- | --- |
| SARLA atlas + guided kernel | **~12 min** (1 A100) | −34.24 | screening posterior |
| guided RWM, 6,400 draws | ~15 min | −58.30 | screening posterior |
| L-BFGS ×16 / Newton | 34 min / 12 min | −217.8 / −2752.6 | point estimates |
| **ADEMCMC, 64 chains** | **32.7 h/chain**, all concurrent | **−25.56** | converged posterior, R̂ < 1.1 |

The JAX path is ~150× faster to an answer and lands ~9 C-log-units short of
where every production chain ends up. That is the screening-grade versus
publication-grade distinction, made concrete on the hardest site we have,
and it is the strongest argument yet for the hybrid we have been proposing
to the CARDAMOM team: **use the JAX modes and Laplace covariance to seed and
shape ADEMCMC rather than to replace it.** ADEMCMC spends ~10⁸ evaluations
per site, ~65 min of it before sampling even starts — both are costs the
fast path can donate away without changing what the sampler produces.

### Does the fast path actually reproduce the posterior? (2026-08-31)

With a converged reference in hand, the screening posterior can finally be
scored rather than assumed. Per-parameter marginal KL(ADEMCMC ‖ SARLA-guided)
at NL-Loo, against a 40×-effort control run:

| metric | 2,000 × 64 | 20,000 × 256 | ADEMCMC floor |
| --- | --- | --- | --- |
| split-half self-KL (Monte-Carlo error) | 0.204 | **0.048** | 0.022 |
| KL vs ADEMCMC | 0.314 | **0.209** | — |
| params with KL < 0.1 | 13/89 | 23/89 | — |
| params with KL > 1 | 16/89 | 9/89 | — |
| between-chain sd ÷ within-chain sd | 3.17 | 2.18 | 0.49 |
| width ratio, 10th pct | 0.53 | 0.57 | 1.0 |

**Half the original gap was noise; the rest is bias that sampling does not
remove.** 40× the effort cut self-inconsistency 4× (to near the reference
floor) but moved the disagreement with truth only 0.314 → 0.209 — now four
times the sampler's own noise floor. Meanwhile chains still visit **1.2
chart regions after 20,000 steps** and the between/within ratio is still
2.18: the walkers never traverse the ridge. The pooled chart weights are
therefore inherited from the *initialization*, and the posterior stays
under-dispersed (10th-percentile width 0.57, unchanged across the 40×
increase).

**Retraction.** Earlier entries on this page described the 12-minute run as
a "screening posterior". That is too strong. It is a good ridge/mode locator
(C-verified best point improved to −190.6 in the long run) and an unreliable
uncertainty product — and uncertainty is what CARDAMOM's science outputs
depend on. The fix is structural, not computational: estimate the
between-chart weights (thermodynamic integration or bridge sampling per
chart, then reweight pooled local draws) instead of expecting a local kernel
to sample them.

### Does the difference matter scientifically? (2026-08-31)

KL is not a science unit, so both posteriors (500 draws each) were pushed
through the C oracle and compared on the quantities CARDAMOM papers report.
"sd" = shift in units of the ADEMCMC posterior standard deviation.

| quantity | ADEMCMC median [5–95%] | fast-path median [5–95%] | shift | CI width |
| --- | --- | --- | --- | --- |
| **NBE** | −1.05 [−1.26, −0.77] | −1.05 [−1.25, −0.82] | **−0.1% (0.0 sd)** | 0.87× |
| GPP | 3.41 [2.91, 4.02] | 3.20 [2.72, 3.83] | −6.4% (−0.6 sd) | 0.99× |
| LAI | 2.08 [1.55, 2.69] | 2.11 [1.66, 2.62] | +1.6% (0.1 sd) | 0.84× |
| wood residence time* | 15.5 yr | 14.9 yr | −3.9% (−0.3 sd) | 0.96× |
| CUE | 0.613 | 0.659 | +7.5% (0.4 sd) | 1.19× |
| ET | 0.156 | 0.209 | +34% (0.4 sd) | 1.14× |
| **SOM residence time** | **689 yr [99, 8750]** | **314 yr [63, 1590]** | **−54%** | **0.18×** |
| CH₄ fraction | 6.1e−4 [~0, 0.38] | 5.0e−3 [~0, 0.53] | +720% (0.0 sd) | 1.40× |

\* wood row uses a crude allocation proxy (0.3·GPP), not the model's wood
turnover flux — indicative only. The SOM row uses actual Rh fluxes.

**Two-sided answer.** For well-constrained carbon fluxes the fast path is
scientifically equivalent: NBE — the headline number of a carbon-balance
study — agrees to 0.1%, and GPP/LAI/CUE agree within 0.6 sd with correct
interval widths. For weakly-constrained quantities it is **biased AND
overconfident**: SOM residence time is 2.2× too short with an interval 5.5×
too narrow. Those are precisely CARDAMOM's signature outputs (cf. Bloom et
al. 2016 PNAS, decadal residence times) — a paper using the fast posterior
would report a confidently wrong soil turnover.

**Why this pattern:** it follows directly from the trapping diagnosis.
Tightly-constrained parameters are pinned by the likelihood *inside* every
chart, so local sampling gets them right. Weakly-constrained parameters need
the full prior-scale ridge to be traversed, which trapped chains never do —
so their spread is set by whichever chart a walker started in.

**Proposed usage rule** (free to compute): trust the fast path for fluxes
and well-identified parameters; do not quote its uncertainties for any
parameter whose posterior width exceeds ~0.5 × its prior width. That ratio
is itself the screening flag.

### Between-chart weights by bridge sampling: exact in 2-D, collapses at 89-D

The trapping diagnosis implies the missing quantity is the *relative mass*
of each chart, which a kernel that cannot move between charts cannot supply.
So estimate it instead of sampling it (`scripts/sarla_evidence.py`):
tessellate into Mahalanobis-nearest-chart cells (disjoint ⇒ Σ Z_k = Z, no
double counting), and bridge-sample each cell between the trapped chains'
draws and that chart's Gaussian restricted to the cell.

**Ground truth (2-D, three modes of known mass 0.5/0.3/0.2, chains unable to
cross):** recovers **0.500 / 0.300 / 0.200** exactly — and gives the same
answer from an adversarial initialization that put 83% of walkers in the
smallest mode. The mechanism is correct.

**NL-Loo (89-D): it makes things much worse.**

| | unweighted | evidence-reweighted |
| --- | --- | --- |
| KL vs ADEMCMC (median) | 0.209 | **1.055** |
| params with KL < 0.1 | 23/89 | 1/89 |
| params with KL > 1 | 9/89 | 50/89 |
| width ratio (median / 10th) | 0.91 / 0.57 | 0.64 / 0.30 |

The proximate cause is visible in the built-in diagnostics: **one cell of 81 absorbs
98.4% of the estimated mass** (mass ESS 1.0, weight ESS 0.012), and 20 cells
received no estimate at all and were implicitly zeroed.

**Split-half stability check: the collapse is estimator variance, not a
property of the cells.** Splitting the pooled draws in half and estimating
each cell's log Z independently from each half (40 cells estimated in both,
1,119 s) gives a **median half-to-half difference of 40 log-units** (90th
percentile 8.7e4, max 7.8e5). Each half is separately degenerate — top-1 mass
0.999 and 1.000, mass ESS 1.0 in both — but they **disagree about which cell
holds the mass** (argmax cell 9 vs cell 5). Two independent samples of the
same estimator putting ~all mass on different cells is the signature of an
estimator whose variance has exploded, not of a genuinely dominant cell. So
the atlas is not demonstrably mis-covering the ridge; the bridge estimator is
simply unusable at this dimension, and the 98.4% figure above should be read
as noise rather than as a measurement.

**This is the third independent failure of density-derived chart weights in
89-D** — after exp(P) weights and Laplace evidence weights — and the most
informative one, because bridge sampling is the most principled of the three
and is provably exact in low dimension. Uniform tiling remains the best
weighting we have. The obstacle looks dimensional rather than
methodological: cell-restricted evidence in 89-D is an integral no local
estimator has yet pinned down.

> **RETRACTED 2026-08-31 (later the same day).** The paragraph above is
> wrong, and its error is instrumental rather than conceptual. Every number
> in this subsection was measured on an atlas in which **24 of 81 charts had
> NaN Hessians** and had silently degraded to prior-width isotropic
> proposals — because the KNORR overflow guard and the `gate='none'`
> curvature mode both existed, both were tested, and neither had been merged
> into `jax-port`. With curvature restored (degraded charts 24 → 4),
> reweighting at 89-D gives KL 0.243–0.679 and width ratios 0.89–0.95, not
> KL ≈ 5 and width 0.19. It is still not a *win* — see
> "Budget, allocation, and the transfer test" below, where it never beats
> the raw sample — but "collapses" was a statement about broken equipment.
> The split-half variance finding immediately above is unaffected: it was
> about estimator stability, and it reproduces.

## A mid-complexity toy with exact ground truth (2026-08-31)

Every diagnosis above compares SARLA to ADEMCMC, so a disagreement never
says which one is wrong. `scripts/toy_mid.py` removes that ambiguity: a
24-D, 6-basin warped Gaussian mixture with a **closed-form density and exact
i.i.d. sampling**, so both samplers are scored against truth, and a full
comparison runs in ~30 s instead of 19 min.

Two design points were needed before it reproduced anything, and both are
findings in their own right.

**Basin widths must be ANTI-correlated with basin mass.** The first version
used random widths, and SARLA recovered the true weights almost exactly
(mass TV 0.042). Multi-start seeding spreads charts in proportion to
*basin-of-attraction volume*, which in that construction happened to track
posterior mass, so the inherited-weight bias did not exist. Making heavy
basins narrow and light basins wide breaks the coincidence. Any surrogate
built without this will silently fail to reproduce the bug.

**The identifiability split has to be designed in.** One direction is stiff
in every basin *and* has the same value in every basin (the NBE analogue);
one is weak and basin-dependent (the residence-time analogue).

It then reproduces the NL-Loo signature:

| | KL vs truth | width med/10th | mass TV | regions/chain |
| --- | --- | --- | --- | --- |
| truth (2nd sample) | 0.000 | 1.00 / 1.00 | 0.003 | — |
| SARLA (guided RWM) | 0.054 | 0.84 / 0.78 | 0.197 | **1.02** |
| DE-MC, matched budget | 0.188 | 0.79 / 0.60 | 0.211 | 1.53 |
| DE-MC, 60× budget | 0.012 | 0.93 / 0.89 | 0.120 | 2.04 |

The **KL ratio of SARLA to the converged reference is 4.5×; at NL-Loo it was
4.4×** (0.209 vs 0.048). The flux column is flat across every method (bias
0.00, CI ratio 1.00), reproducing "NBE agrees no matter who sampled it".

Caveats: the DE-MC here is a reimplementation of the ADEMCMC *family*, not
CARDAMOM's C sampler, and it needs a multi-start mode-search init (the
analogue of the EDC search) or it never populates the narrow basins at all.
Its max R-hat is 1.86 even at 60× budget, so it is a behavioural stand-in,
not a converged reference — truth is the reference.

### What the toy found: coverage, not weighting

Substituting exact π-draws for the trapped chains' draws inside each cell,
changing nothing else, moves mass TV from **0.555 to 0.034** against a 0.028
exact-logZ floor. The bridge estimator is sound; its documented-but-untested
input assumption is false. Two competing explanations were tested and both
refuted: keep fractions are healthy (median 0.984) so the tessellation is
fine, and the 7 unestimated cells hold 0.019 of the mass so dropping them is
immaterial.

The scan also exposed a real defect: every `logZ` carried a constant −26.37
offset, exactly 24·ln 3 = Σ log(scale). The bridge runs in whitened space
while the target was evaluated in z-space, with no Jacobian. It cancels in
`reweight()`, so no conclusion changed, but `logZ` was not an evidence.
Fixed; the 2-D toy now returns log 0.5 / log 0.3 / log 0.2 exactly.

**Five mechanisms were then tried to lift coverage, and all five failed:**

| mechanism | outcome |
| --- | --- |
| cell-restricted Metropolis | correct by construction, but small cells never equilibrate (203 seeds: radius 0.52 → 0.64 at 8000 steps) |
| coarse (per-basin) tessellation | exact given good draws (logZ error 0.01); useless given trapped ones |
| atlas-wide independence jumps | 0.002 acceptance |
| region-local independence jumps | **0.008** acceptance — the destination set was never the problem; the Laplace charts are the wrong *shape* |
| region-DE (ADEMCMC's mechanism) | rank-deficient below n≈d chains/region (coverage 0.04–0.08 at 10/region); never beats chart-RWM even at 48/region |

The one thing that worked was not clever: **allocate chains per region, not
per chart**. Chains were being handed out in proportion to charts-per-region,
so the largest basin got ~35 and the smallest got **1**. Fixing that raised
worst-basin coverage 10× and took the width ratio from 0.84/0.78 to
1.02/0.94 — the posterior-width compression, gone, at zero cost.

Also corrected here: region-DE refuted my premise that DE-MC's advantage is
better within-basin preconditioning. It is not. DE-MC wins because its
chains span *all* basins, so its difference vectors supply cross-basin
moves; confined to one basin it has no edge.

### Budget: the answer was mostly compute

Scaling the best configuration 1×/4×/16×/64×:

| arm | KLraw | wRaw | **KLrewt** | **wRewt** | TVrewt | worst coverage | evals |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ×1 | 0.042 | 1.02 | 0.115 | 0.69 | 0.300 | 0.20 | 0.19M |
| ×4 | 0.036 | 1.11 | 0.049 | 0.80 | 0.229 | 0.37 | 0.77M |
| ×16 | 0.034 | 1.14 | 0.026 | 0.84 | 0.151 | 0.50 | 3.07M |
| ×64 | 0.035 | 1.15 | **0.020** | **0.86** | **0.143** | 0.67 | 12.29M |
| ADEMCMC converged | | | 0.012 | 0.93 | 0.120 | | 11.5M |

Nothing saturates. At matched cost SARLA reaches KL 0.020 / width 0.86 /
TV 0.143 against ADEMCMC's 0.012 / 0.93 / 0.120 — close, still improving
where ADEMCMC has converged.

**A measurement error worth recording.** I first read KL and width off the
*raw* sample and reported that KL saturates at 0.034. Balanced allocation
distorts the raw mixture weights *on purpose* — that is the whole point of
decoupling exploration from mass — so the raw sample was never the thing to
score. Reweighted, KL falls 5.8× and does not saturate. Aggregate statistics
on a deliberately-distorted sample are not a result.

**Operational rule.** Reweighting *hurts* below a coverage crossover
(×1: 0.042 → 0.115) and helps above it (×16: 0.034 → 0.026). NL-Loo's
12-minute run sat well below that crossover, which is why every reweighting
attempt there made things worse. The method was being used outside its
regime.

## Budget, allocation, and the transfer test at 89-D (2026-08-31)

First attempt said the toy does not transfer. It was measured on an atlas
with 24/81 charts blind (see the retraction above). After merging both
fixes — `cc52936e` (overflow guard) and `f93aa7a9` (`gate='none'`
curvature), equivalence suite 44 passed / 1 skipped — the atlas rebuilt with
**4/81 degraded**, and all six arms were rerun with the per-chart control
the first attempt lacked.

| arm | KLraw | wRaw | w10 | KLrewt | wRewt | reg/ch | best P |
| --- | --- | --- | --- | --- | --- | --- | --- |
| bal ×1 | 0.460 | 1.19 | 0.78 | 0.661 | 0.95 | 1.00 | −213.60 |
| bal ×4 | 0.322 | 1.12 | 0.82 | 0.535 | 0.92 | 1.12 | −213.41 |
| bal ×16 | **0.226** | 1.11 | **0.83** | 0.268 | 0.93 | 1.12 | **−204.65** |
| perchart ×1 | 0.633 | 0.97 | 0.60 | 0.679 | 0.89 | 1.00 | −214.46 |
| perchart ×4 | 0.376 | 0.97 | 0.59 | 0.438 | 0.91 | 1.00 | −207.56 |
| perchart ×16 | 0.231 | 0.96 | 0.65 | 0.243 | 0.91 | 1.00 | −205.37 |

**Transfers.** Budget helps monotonically at 89-D for both allocations
(0.460 → 0.226 and 0.633 → 0.231), no plateau. Balanced allocation fixes the
width compression exactly as in the toy: 10th-percentile width 0.60 → 0.83
(toy 0.78 → 0.94). Per-chart reproduces the historical narrow tail
(0.59–0.65) — the mechanism behind the too-tight residence-time interval,
and free to fix.

**Does not transfer.** The density clustering degenerates here: region sizes
`[1 1 1 69 2 1 3 1 1 1]`, byte-identical before and after the curvature fix,
so it is a genuine property of this posterior and not an artifact. SARLA is
*over*-dispersed at 89-D (coverage 3.5–6.3× vs ADEMCMC), the opposite of the
toy's under-coverage, and unexplained. Chains still do not change region
(reg/ch 1.00–1.12). Reweighting never beats raw here, though its penalty
shrinks with budget (bal +0.201, +0.213, +0.042), so the crossover is
directional but beyond 16×.

**Caveat on cross-run comparison.** These KLs are binned in whitened space;
the historical 0.209 was binned in log-parameter space. Histogram KL is not
invariant under a change of variable, so **no claim** is made against 0.209.
The six arms are internally comparable; the cross-run number is not.

**Newly visible:** 5 of 81 chart centres are infeasible under the hard gate
— the original atlas planted charts outside the EDC-feasible set. The
builder reports this but does not yet enforce it, so those 5 receive ungated
curvature that is strictly not the posterior's.

### What the day changed, and the process lesson

The recipe that survives is unglamorous: **allocate chains per region, and
spend more compute.** Both were available on day one. Against that, five
increasingly elaborate mechanisms were built and all five failed.

The sharper lesson is about order of operations. A full hour was spent
building an argument for why the toy's mechanism could not survive at 89-D —
dimensionality, geometry, cell structure — when the actual cause was that a
third of the atlas had no curvature, from a fix already written, already
verified, and merely unmerged. Theorising about a surprising result before
checking whether the instruments work is the wrong order, and it cost a full
set of conclusions that then had to be withdrawn.

### Still open

- Localize the third overflow site: 4 of 81 charts still have NaN Hessians
  after both fixes (was 24), consistent with the unlocalized anchor-79 site.
- Enforce, not merely report, hard-gate feasibility before taking `gate='none'`
  curvature; and decide what to do about the 5 infeasible chart centres.
- Explain the 89-D over-dispersion (3.5–6.3× vs ADEMCMC). It is the one
  symptom the toy gets backwards, so the toy cannot be used to chase it.
- A mixing mechanism that moves between regions. Six have now failed;
  parallel tempering across the atlas remains untried.
- Re-run Newton arm C now that curvature is usable — H100 preferred
  (10× cheaper Newton steps there).
- Seeded-ADEMCMC experiment: JAX modes + Laplace covariance as the C
  sampler's starting state and proposal, measuring how much of the 32.7 h
  disappears with the product held fixed.
- Revisit the science-impact table: its caveat should become a budget
  threshold ("trust the fast path for weakly-constrained quantities above
  N evaluations") rather than a blanket prohibition.

---

## Context

- The pilot pipeline and its measured numbers: [FINDINGS.md](../FINDINGS.md)
- Concept figures: [CONCEPTS.md](CONCEPTS.md)
- SARLA design note: https://app.notion.com/p/3cd788b60530817a904ff70d7f6ef58b
- Toy target and harness: `scripts/toy_mid*.py`; NL-Loo transfer test:
  `scripts/nlloo_budget.py`, `scripts/nlloo_build_charts.py` (research repo)
- This page exists because the defect was caught by a reader's question
  about a figure — the fourth time in this project that adversarial
  attention to a surprising detail overturned a headline number. The
  ADEMCMC control is the fifth: it was run only because a reader asked
  "how well does ADEMCMC do on NL-Loo?", and it reversed a conclusion this
  page had already drawn. The 2026-08-31 retraction above is the sixth, and
  the only one so far caused by unmerged code rather than by reasoning: the
  fix that overturned it had been written, verified, and left on a branch.
