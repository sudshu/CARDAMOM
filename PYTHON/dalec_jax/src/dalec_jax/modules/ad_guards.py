"""Tangent-space guards for gradient-hardened expressions.

These helpers reproduce their C-matching primal bit-exactly AS AN EXPRESSION
(verified over a 400,013-point sweep spanning both cutoffs: zero differing
bits) and only change how derivatives are computed, fixing a NaN-Hessian
leak:

    w = 1 / jnp.exp(jnp.minimum(x, LOG_DBL_MAX))

For x just below the cutoff (~702.6 .. 709.78], exp(x) is a finite huge
value (1e305..DBL_MAX) and w is a selected tiny value. The raw JVP of exp
forms exp(x) * dx, which overflows to inf whenever |dx| >= DBL_MAX/exp(x)
(as small as ~7.35 here); the subsequent div JVP then yields inf/inf = NaN.
Reverse-mode never forms exp(x)*dx (its cotangent chain divides by
exp(x)^2 = inf, underflowing benignly to 0), so plain jax.grad stays finite
while any forward pass over the graph (jvp, jacfwd(grad), HVPs) goes NaN.
Localized at DALEC_1100 site NL-Loo: x_gf = 707.7897 at timestep 86 (the
ALLOC growth factor), NaN Hessians in the 12 KNORR/labile directions.

A jax.custom_jvp wrapper is NOT sufficient: inside lax.scan the
forward-over-reverse pass (jvp of grad) differentiates the scan body after
linearization has decomposed the custom call, so the raw 1/exp body gets
re-differentiated and the overflow returns (verified with a single-step
scan micro-repro; eager HVPs were fine, scan HVPs NaN'd).

Instead the guard makes the BODY itself safe to differentiate at any order
via the straight-through idiom (m = min(x, C), sg = stop_gradient):

    out = sg(1/exp(m)) + where(isfinite(exp(-m)), exp(-m) - sg(exp(-m)), 0)

- Primal value: exp(-m) - sg(exp(-m)) is exactly +0.0 (same float minus
  itself), so out == 1/exp(m) bit-exactly — unconditionally, even under
  the FTZ/DAZ subnormal flushing XLA CPU applies (which breaks the
  compensated-sum variant exp(-m) + sg(1/exp(m) - exp(-m)) by flushing
  the 1-ulp subnormal correction for x in ~(672.4, 709.78]). The
  isfinite(exp(-m)) guard covers m < ~-709.78 (x very negative, reachable
  in the module goldens), where exp(-m) = inf would make the pair
  inf - inf = NaN while the C computes 1/exp(m); there the correction is
  0 and out is the C value, with a zero (rather than the old NaN) tangent.
- Every derivative order flows through exp(-m) only, whose JVP is
  -exp(-m)*dm — no huge intermediate exists on any AD path.

MEASURED SCOPE AND LIMITS (81 SARLA chart centers at NL-Loo x the 12
implicated parameter directions = 972 HVPs, equivalence-grade flags
XLA_FLAGS=--xla_disable_hlo_passes=algsimp):
- NaN HVP directions 163 -> 12; anchors with any NaN 21 -> 1.
- INCOMPLETE: anchor 79 retains all 12 NaN directions, unchanged by this
  guard => at least one FURTHER overflow site exists elsewhere in the step.
  Not yet localized; the same bisection recipe applies.
- Whole-model primal is bit-identical at 80/81 anchors; one anchor shifts
  by 5.7e-14 absolute (~1-2 ulp on a log-posterior near -200, i.e. ~1e5
  inside the port's 1e-10 trajectory bar). The guard is exact in isolation,
  so the shift comes from XLA fusing the enlarged graph differently, not
  from this arithmetic. Under DEFAULT flags (algsimp on) 4/77 anchors shift
  by the same magnitude. Do not describe this change as globally
  bit-preserving without re-measuring.
"""
import jax
import jax.numpy as jnp

LOG_DBL_MAX = 709.782712893384  # exp overflows to inf strictly above


def inv_exp_clamped(x):
    """Bit-exact 1 / exp(min(x, LOG_DBL_MAX)) with overflow-free autodiff
    at every order (safe inside lax.scan; see module docstring)."""
    m = jnp.minimum(x, LOG_DBL_MAX)
    w_stable = jnp.exp(-m)                    # derivative path (bounded above)
    w_exact = 1 / jnp.exp(m)                  # value path (C-exact bits)
    sg = jax.lax.stop_gradient
    zero_pair = jnp.where(jnp.isfinite(w_stable),
                          w_stable - sg(w_stable), 0.0)   # +0.0 exactly
    return sg(w_exact) + zero_pair
