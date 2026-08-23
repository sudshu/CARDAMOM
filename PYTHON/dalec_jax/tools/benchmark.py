#!/usr/bin/env python3
"""P6 benchmark + gradient hygiene scan.

Usage:
  env -u LD_LIBRARY_PATH python tools/benchmark.py            # GPU if visible
  env -u LD_LIBRARY_PATH JAX_PLATFORMS=cpu python tools/benchmark.py

Reports:
  - compile + steady-state wall time for vmap'd forward model and full
    pipeline (model+EDCs+likelihood) at several batch sizes
  - per-trajectory time vs the C baseline (0.196 ms single-thread, measured
    in cardamom_research notes/benchmark.md)
  - jax.grad(P) NaN/Inf scan at 64 genuinely-running posterior samples
    (insurance for the gradient-based-inference phase; NOT an equivalence
    gate — see plan.md P6)
"""
from __future__ import annotations

import os
import time

import numpy as np

import dalec_jax  # noqa: F401  (x64 on)
import jax
import jax.numpy as jnp

from dalec_jax import edcs, oracle_io
from dalec_jax.likelihood import data_prep, mlf2
from dalec_jax.model import prederive_vegk, run_dalec_1100

C_BASELINE_MS = 0.196


def main() -> int:
    dev = jax.devices()[0]
    print(f"jax {jax.__version__} | device: {dev.device_kind} ({dev.platform})"
          f" | x64: {jnp.zeros(1).dtype == jnp.float64}")

    g = oracle_io.trajectory_golden()
    cbf = data_prep.load_cbf(g["meta"]["cbf"])
    edc_cfg = {"n_timesteps": cbf.n_timesteps,
               "dint": edcs.compute_dint(cbf.time),
               "edc_eqf": cbf.edc_eqf, "skt_ref_mean": cbf.skt_ref_mean}
    VegK = prederive_vegk(cbf.met["DOY"], cbf.LAT)
    pars = np.fromfile(oracle_io.GOLDEN / "posterior/params.bin").reshape(-1, 89)
    gold_P = np.fromfile(oracle_io.GOLDEN / "posterior/mlf.bin").reshape(
        pars.shape[0], -1)[:, -1]

    def fwd(p):
        return run_dalec_1100(p, cbf.met, cbf.LAT, cbf.deltat, VegK)

    def full_P(p):
        pools, fluxes = fwd(p)
        _, _, P = mlf2(cbf, edc_cfg, p, pools, fluxes)
        return P

    print(f"\n{'batch':>6} {'kind':10s} {'compile(s)':>10} {'run(ms)':>10} "
          f"{'ms/traj':>9} {'vs C 1-thread':>13}")
    for B in (128, 1024, 4000):
        batch = jnp.asarray(pars[:B])
        for kind, fn in (("forward", jax.jit(jax.vmap(fwd))),
                         ("full P", jax.jit(jax.vmap(full_P)))):
            t0 = time.time()
            out = fn(batch)
            jax.tree_util.tree_map(lambda x: x.block_until_ready(), out)
            t_compile = time.time() - t0
            reps = 3
            t0 = time.time()
            for _ in range(reps):
                out = fn(batch)
                jax.tree_util.tree_map(lambda x: x.block_until_ready(), out)
            dt = (time.time() - t0) / reps
            per = dt * 1e3 / B
            print(f"{B:>6} {kind:10s} {t_compile:>10.2f} {dt*1e3:>10.1f} "
                  f"{per:>9.4f} {C_BASELINE_MS/per:>12.1f}x")

    # ---- gradient hygiene scan (64 genuinely-running posterior samples)
    genuine = np.where(np.isfinite(gold_P))[0]
    sel = genuine[np.linspace(0, len(genuine) - 1, 64).astype(int)]
    gradP = jax.jit(jax.vmap(jax.grad(full_P)))
    t0 = time.time()
    Gs = np.asarray(gradP(jnp.asarray(pars[sel])))
    t_grad = time.time() - t0
    finite = np.isfinite(Gs).all(axis=1)
    print(f"\ngradient scan: {finite.sum()}/64 samples with fully finite "
          f"dP/dpars (89 components); wall {t_grad:.1f}s incl. compile")
    if not finite.all():
        bad = np.where(~finite)[0]
        from dalec_jax.indices import PAR_NAMES
        counts = (~np.isfinite(Gs)).sum(axis=0)
        worst = np.argsort(counts)[::-1][:10]
        print("  non-finite gradient components (top 10 by frequency):")
        for k in worst:
            if counts[k]:
                print(f"    {PAR_NAMES[k]:24s} non-finite in {counts[k]}/64")
    gmag = np.abs(Gs[finite])
    if gmag.size:
        print(f"  finite-sample |grad| median {np.median(gmag):.3g}, "
              f"max {gmag.max():.3g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
