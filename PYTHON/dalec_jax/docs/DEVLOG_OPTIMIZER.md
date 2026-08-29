# Devlog: is the "MAP" actually the MAP? L-BFGS vs true Newton

A running record of an open investigation. Status: **smoke test running**.
Numbers on this page are measured; anything not yet measured is marked TBD.

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

### Full comparison: TBD (running)

### Decision: TBD

Possible outcomes and what each would mean for the pipeline:

- **B ≈ C ≫ A**: iteration budget, not method — raise the L-BFGS budget.
- **C ≫ B**: curvature staleness is real — adopt Newton polish (arm D)
  as a standard stage between L-BFGS and the Hessian/Laplace step.
- **Nothing beats the chain**: the ridge is effectively flat along its
  length; "the MAP" is numerically ill-defined here and only the sampler's
  mass answer is meaningful.

## Context

- The pilot pipeline and its measured numbers: [FINDINGS.md](../FINDINGS.md)
- Concept figures: [CONCEPTS.md](CONCEPTS.md)
- This page exists because the defect was caught by a reader's question
  about a figure — the fourth time in this project that adversarial
  attention to a surprising detail overturned a headline number.
