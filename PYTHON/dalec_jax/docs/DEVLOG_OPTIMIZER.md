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

### Smoke (4 starts, 15 Newton iters — machinery check only): TBD

### Full comparison: TBD

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
