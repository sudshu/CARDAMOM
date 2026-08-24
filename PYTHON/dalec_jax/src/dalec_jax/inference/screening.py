"""EDC-feasible starting points by batched iid prior rejection.

Feasible vectors are rare but *quantifiably* rare: the measured full-gate
pass rate at the demo site is 5.7e-7 (6 / 10,485,760 iid draws), audit-
replicated at 7.6e-7 with every hit confirmed finite through the C oracle.
Blind rejection therefore needs ~1.3-1.8M full-pipeline evaluations per
start: ~20 s on one A100, ~20 min on one C core. This is bounded and
trivially parallel — unlike an annealed local search, it cannot stall
(the single-chain MCMCID-119 EDC search never terminated at this site in
two attempts).

Note: hits pass the gate but are typically POOR fits (P <= -3.8e4 at the
demo site); chains seeded from them still need normal burn-in. Optimizer
modes (see laplace.py) are far better seeds when available.
"""
from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np


def find_feasible_starts(logpost, n_starts: int, seed: int = 0,
                         batch: int = 16384, max_draws: int = 200_000_000,
                         verbose: bool = True):
    """Draw u ~ U[0,1]^89 iid until n_starts pass the full gate.

    Returns (z_hits (n,89) ndarray, P_hits (n,), n_evaluated). The gate is
    isfinite(logpost(z)) — model + 15 EDCs + finite likelihood.
    """
    eval_chunk = jax.jit(jax.vmap(logpost))
    key = jax.random.PRNGKey(seed)
    z_hits, p_hits, n_eval = [], [], 0
    t0 = time.time()
    while sum(len(h) for h in z_hits) < n_starts and n_eval < max_draws:
        key, k = jax.random.split(key)
        u = jax.random.uniform(k, (batch, 89), dtype=jnp.float64)
        u = jnp.clip(u, 1e-12, 1 - 1e-12)
        z = jnp.log(u) - jnp.log1p(-u)
        lp = np.asarray(eval_chunk(z))
        n_eval += batch
        ok = np.isfinite(lp)
        if ok.any():
            z_hits.append(np.asarray(z)[ok])
            p_hits.append(lp[ok])
            if verbose:
                n = sum(len(h) for h in z_hits)
                print(f"  [screen] {n}/{n_starts} hits after {n_eval:,} "
                      f"draws ({time.time()-t0:.0f}s)", flush=True)
    z_all = (np.concatenate(z_hits) if z_hits
             else np.empty((0, 89)))[:n_starts]
    p_all = (np.concatenate(p_hits) if p_hits else np.empty(0))[:n_starts]
    if verbose:
        rate = max(len(z_all), 1) / max(n_eval, 1)
        print(f"  [screen] done: {len(z_all)} starts / {n_eval:,} draws "
              f"(rate ~{rate:.1e}), {time.time()-t0:.0f}s", flush=True)
    return z_all, p_all, n_eval
