"""Vmapped random-walk Metropolis with an arbitrary proposal covariance.

Measured at the demo site (four proposal shapes raced, then adversarially
audited with a grid-tuned re-race and a same-start control):
- the capped-Laplace covariance (see laplace.cap_covariance) was the best
  shape tested — 2.4-3x the mixing increment of a per-parameter diagonal,
  matching the "oracle" full chain covariance on median-parameter mixing
  (the oracle keeps the better worst-parameter mixing);
- but NO shape rescues plain RWM on this target: >= ~2e4 iterations per
  effective sample even for the best shape (median parameter).

CAUTION on diagnostics: multi-chain bulk ESS from short runs measures
ensemble DIVERSITY, not converged effective samples — an audited control
with all chains started at ONE point scored HIGHER ESS than the legitimate
run while demonstrably biased (R-hat 1.96). Before believing a short-run
ESS, run a same-start control and check split-R-hat.
"""
from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np


def run_rwm(logpost, z0, proposal_cov, n_iters: int = 20000,
            scale: float | None = None, thin: int = 10, chunk: int = 500,
            seed: int = 0, verbose: bool = True):
    """Run len(z0) parallel Metropolis chains.

    proposal step: z' = z + scale * chol(proposal_cov) @ xi. Default scale
    is the Roberts-Rosenthal 2.38/sqrt(d); tune toward ~15-30% acceptance
    (at the demo site the informed shapes mixed BETTER at somewhat larger,
    lower-acceptance scales than the 23% rule suggests).

    Returns dict: z (chains, n_stored, d), acc (scalar), scale.
    """
    d = np.asarray(z0).shape[1]
    if scale is None:
        scale = 2.38 / np.sqrt(d)
    cov = np.asarray(proposal_cov)
    # A non-finite proposal covariance does NOT raise downstream: the
    # Cholesky yields NaN, every proposal is NaN, every Metropolis ratio
    # compares False, and the chains sit at their seeds with 0% acceptance
    # looking like a merely hard target. Observed on FluxVal site 71, where
    # the exact Hessian at a cliff-side mode came back non-finite. Fail
    # loudly instead.
    if not np.isfinite(cov).all():
        raise ValueError(
            "proposal_cov contains non-finite entries; the chains would "
            "freeze silently at 0% acceptance. The usual cause is an exact "
            "Hessian taken at a mode sitting on an EDC cliff — pick a mode "
            "whose covariance is finite and positive definite (see "
            "evidence_weights, which gives unusable modes weight 0).")
    L = jnp.asarray(np.linalg.cholesky(cov + 1e-12 * np.eye(d)) * scale)

    def one_iter(carry, key):
        z, lp, acc = carry
        k1, k2 = jax.random.split(key)
        zp = z + L @ jax.random.normal(k1, (d,))
        lpp = logpost(zp)
        take = jnp.log(jax.random.uniform(k2)) < (lpp - lp)
        return (jnp.where(take, zp, z), jnp.where(take, lpp, lp),
                acc + take), None

    n_emit = chunk // thin

    @jax.jit
    @jax.vmap
    def run_chunk(z, lp, key):
        def emit(carry, keys):
            carry, _ = jax.lax.scan(one_iter, carry, keys)
            return carry, carry[0]
        keys = jax.random.split(key, chunk).reshape(n_emit, thin, -1)
        (z, lp, acc), zs = jax.lax.scan(emit, (z, lp, 0.0), keys)
        return z, lp, acc / chunk, zs

    z = jnp.asarray(z0)
    lp = jax.jit(jax.vmap(logpost))(z)
    key = jax.random.PRNGKey(seed)
    stored, accs = [], []
    t0 = time.time()
    for c in range(n_iters // chunk):
        key, k = jax.random.split(key)
        ks = jax.random.split(k, z.shape[0])
        z, lp, acc, zs = run_chunk(z, lp, ks)
        stored.append(np.asarray(zs)); accs.append(float(np.mean(acc)))
        if verbose and (c + 1) % max((n_iters // chunk) // 5, 1) == 0:
            print(f"  [rwm] iter {(c+1)*chunk}: acc "
                  f"{np.mean(accs)*100:.1f}% ({time.time()-t0:.0f}s)",
                  flush=True)
    return {"z": np.concatenate(stored, axis=1),
            "acc": float(np.mean(accs)), "scale": float(scale)}
