"""Inference fast path for DALEC_1100, built on the port's exact gradients.

The pieces (each measured and adversarially audited on the demo site; see
README "Inference fast path" and the research archive's PERFORMANCE.md /
AUDIT reports for the numbers):

- :func:`build_logpost` — the z = logit(u) log-posterior, the SAME target
  the C sampler uses (uniform prior on normalized u, log-ratio parameter
  mapping, transform Jacobian included).
- :func:`find_feasible_starts` — batched iid prior rejection through the
  full gate (model + 15 EDCs + finite likelihood). Measured pass rate at
  the demo site ≈ 6e-7, i.e. ~1.8M evaluations (~20 s on one A100) per
  start.
- :func:`multipoint_laplace`, :func:`exact_hessians`,
  :func:`cap_covariance`, :func:`evidence_weights` — multipoint L-BFGS
  optimization, exact 89x89 Hessians, and the prior-width repair of flat
  curvature directions (spread ratio vs MCMC 2.22 -> 0.92 median at the
  demo site).
- :func:`run_rwm` — vmapped random-walk Metropolis with an arbitrary
  proposal covariance; feeding it the capped Laplace covariance was the
  best of four shapes tested (2.4-3x the mixing of per-parameter tuning).

Requires the optional dependency ``optax`` (``pip install "dalec-jax[inference]"``).
"""
from .target import build_logpost, logit_jacobian, nor2par, par2nor
from .screening import find_feasible_starts
from .laplace import (cap_covariance, dedupe_modes, evidence_weights,
                      exact_hessians, mixture_draws, multipoint_laplace)
from .rwm import run_rwm

__all__ = [
    "build_logpost", "logit_jacobian", "nor2par", "par2nor",
    "find_feasible_starts",
    "multipoint_laplace", "exact_hessians", "cap_covariance",
    "evidence_weights", "dedupe_modes", "mixture_draws",
    "run_rwm",
]
