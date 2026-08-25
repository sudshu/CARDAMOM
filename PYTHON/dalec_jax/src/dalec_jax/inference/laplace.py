"""Multipoint Laplace: L-BFGS modes + exact Hessians + prior-width repair.

Measured at the demo site (adversarially audited, C-oracle re-scored):
- 15 of 16 modes from two independent runs sit AT OR ABOVE the best sample
  the production MCMC ever stored; no chain sample within 2.8 log-units of
  the best mode. The MAP-minus-chain-best gap is a cheap per-site
  convergence audit.
- The raw Gaussian overstated posterior spreads ~2.2x (median) because
  near-flat Hessian directions were PSD-floored; capping those directions
  at the PRIOR's width (logistic variance pi^2/3 in z) repairs it to a
  median spread ratio of 0.92 vs the MCMC (|dmean| 0.035 in u-space).
- 32 starts suffice: 256 starts gained +0.36 log-units and the same basin.
- Do NOT importance-reweight the mixture in 89-D: measured collapse
  (2.3% of draws EDC-feasible, weight ESS = 1 of 65,536).

Implementation notes: the L-BFGS scan is CHUNKED — compiling a monolithic
400-iteration optax lbfgs+zoom scan took ~45 minutes of XLA time; chunks
of ~20 compile in seconds. Requires the optional dependency ``optax``.
"""
from __future__ import annotations

import numpy as np

import jax
import jax.numpy as jnp

_PRIOR_VAR = float(np.pi ** 2 / 3.0)   # logistic prior variance in z


def multipoint_laplace(logpost, z0, max_iters: int = 400, chunk: int = 20,
                       verbose: bool = True):
    """Vmapped L-BFGS ascent on logpost from starts z0 (n, 89).

    Returns dict with z_end (n,89), P_end (n; z-space density — includes
    the logit Jacobian, see target.py caution), gnorm_end (n).

    z_end/P_end are the BEST FINITE iterate each chain visited, tracked
    inside the scan — guaranteed consistent (P_end == logpost(z_end)) and
    immune to a final line-search step that walks off a cliff or NaNs.
    (An earlier version returned post-update z with the pre-update value:
    off by one step, and a NaN'd last step could pair a finite P with a
    NaN z, poisoning downstream Hessians/proposals.) gnorm_end is the
    |grad|_inf at the last evaluated iterate — a stationarity diagnostic,
    not necessarily at z_end.
    """
    try:
        import optax
    except ImportError as e:                       # pragma: no cover
        raise ImportError(
            "multipoint_laplace requires optax: pip install "
            "'dalec-jax[inference]'") from e

    neg = lambda z: -logpost(z)
    opt = optax.lbfgs()
    vg = jax.value_and_grad(neg)

    def step(carry, _):
        z, st, zb, vb = carry
        v, g = vg(z)
        better = jnp.isfinite(v) & (v < vb)
        zb = jnp.where(better, z, zb)
        vb = jnp.where(better, v, vb)
        upd, st = opt.update(g, st, z, value=v, grad=g, value_fn=neg)
        z = optax.apply_updates(z, upd)
        return (z, st, zb, vb), (v, jnp.max(jnp.abs(g)))

    @jax.jit
    @jax.vmap
    def run_chunk(z, st, zb, vb):
        (z, st, zb, vb), (vals, gn) = jax.lax.scan(
            step, (z, st, zb, vb), None, length=chunk)
        return z, st, zb, vb, vals, gn

    z = jnp.asarray(z0)
    st = jax.vmap(opt.init)(z)
    zb, vb = z, jnp.full(z.shape[0], jnp.inf)
    g_last = None
    for c in range(max(max_iters // chunk, 1)):
        z, st, zb, vb, vals, gn = run_chunk(z, st, zb, vb)
        g_last = np.asarray(gn[:, -1])
        if verbose:
            with np.errstate(all="ignore"):
                print(f"  [lbfgs] iter {(c+1)*chunk:4d}: best P "
                      f"{-np.nanmin(np.asarray(vb)):.2f} "
                      f"({int(np.isfinite(np.asarray(vb)).sum())}"
                      f"/{len(np.asarray(vb))} finite), median |grad| "
                      f"{np.nanmedian(g_last):.2e}", flush=True)
    # fold in the final iterate (its value was never scanned)
    v_fin = np.asarray(jax.jit(jax.vmap(neg))(z))
    zb, vb = np.array(zb), np.array(vb)     # copy: jax arrays are read-only
    take = np.isfinite(v_fin) & (v_fin < vb)
    zb[take], vb[take] = np.asarray(z)[take], v_fin[take]
    return {"z_end": zb, "P_end": -vb, "gnorm_end": g_last}


def dedupe_modes(z_end, P_end, tol: float = 0.15, max_modes: int = 8):
    """Keep the highest-P representative of each distinct mode (sup-norm)."""
    order = np.argsort(-np.asarray(P_end))
    keep = []
    for i in order:
        if not np.isfinite(P_end[i]):
            continue
        if all(np.max(np.abs(z_end[i] - z_end[j])) > tol for j in keep):
            keep.append(int(i))
        if len(keep) >= max_modes:
            break
    return keep


def exact_hessians(logpost, z_modes, verbose: bool = True):
    """Exact 89x89 Hessians of -logpost at each mode (forward-over-reverse).

    ~30 s per mode on CPU or one A100 at batch 1 (the sequential 240-step
    scan dominates); vmap/chunk across modes if you have many.
    """
    hess = jax.jit(jax.jacfwd(jax.grad(lambda z: -logpost(z))))
    out = []
    for i, zm in enumerate(np.asarray(z_modes)):
        out.append(np.asarray(hess(jnp.asarray(zm))))
        if verbose:
            print(f"  [hessian] {i+1}/{len(z_modes)}", flush=True)
    return np.array(out)


def cap_covariance(H, cap: float = _PRIOR_VAR):
    """Hessian -> covariance with likelihood-flat directions capped at the
    prior's width (the one-line repair; NOT an arbitrary PSD floor).

    Eigen-directions with curvature below 1/cap (including any negative
    curvature at a non-converged iterate) get variance = cap, i.e. the
    prior's own variance in z: "if the data don't curve the surface, your
    uncertainty is the prior's width."
    """
    Hs = 0.5 * (H + H.T)
    w, V = np.linalg.eigh(Hs)
    var = np.where(w > 1.0 / cap, 1.0 / np.maximum(w, 1e-300), cap)
    return (V * var) @ V.T


def evidence_weights(P_modes, covs):
    """Laplace evidence weights over modes (softmax of P + 0.5*logdet).

    Modes whose covariance is not usable get weight 0 rather than
    poisoning the whole vector. This is not hypothetical: the exact
    Hessian can come back non-finite at a mode sitting on an EDC cliff
    (second derivatives through a `where` with a -inf branch), and a
    single NaN logdet used to make `logw.max()` NaN and hence EVERY
    weight NaN. Downstream that silently degraded to "argmax picks mode
    0" — which happened to be the best mode, so it looked like it worked.

    Returns uniform weights only if no mode is usable, so callers always
    get something that sums to 1.
    """
    logw = np.full(len(covs), -np.inf)
    for i, (P, C) in enumerate(zip(P_modes, covs)):
        C = np.asarray(C)
        if not np.isfinite(P) or not np.isfinite(C).all():
            continue
        sign, logdet = np.linalg.slogdet(C)
        if sign <= 0 or not np.isfinite(logdet):
            continue
        logw[i] = P + 0.5 * logdet
    if not np.isfinite(logw).any():
        return np.full(len(covs), 1.0 / len(covs))
    w = np.exp(logw - logw[np.isfinite(logw)].max())
    return w / w.sum()


def mixture_draws(rng, z_modes, covs, weights, n: int):
    """Draw n samples from the Gaussian mixture (screening use only —
    remember it is a symmetric bell on a skewed posterior; ~±35% per-param
    width accuracy after capping, at the demo site)."""
    counts = rng.multinomial(n, np.asarray(weights, dtype=float))
    draws = [rng.multivariate_normal(np.asarray(z_modes[k]), covs[k], size=c)
             for k, c in enumerate(counts) if c]
    z = np.concatenate(draws)
    rng.shuffle(z)
    return z
