#!/usr/bin/env python3
"""End-to-end demo of the Laplace fast path on the bundled demo site.

Pipeline: feasible starts (bundled posterior draws, or --screen for blind
prior rejection) -> multipoint L-BFGS -> distinct modes -> exact Hessians
-> prior-width-capped covariances -> evidence weights -> mixture summary.

Typical run (one A100, defaults):   ~10-15 min, dominated by L-BFGS.
CPU works too (set JAX_PLATFORMS=cpu); expect ~3-5x longer.
--screen replaces the bundled starts with blind rejection (~20 s/start on
an A100 at the measured 6e-7 pass rate; not recommended on CPU).

Run from the package root (PYTHON/dalec_jax):
  env -u LD_LIBRARY_PATH XLA_FLAGS=--xla_disable_hlo_passes=algsimp \
      python examples/laplace_fast_path.py --starts 8 --iters 200
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

import dalec_jax  # noqa: F401  (enables float64)
from dalec_jax.inference import (build_logpost, cap_covariance,
                                 dedupe_modes, evidence_weights,
                                 exact_hessians, find_feasible_starts,
                                 logit_jacobian, mixture_draws,
                                 multipoint_laplace, par2nor)

PKG = Path(__file__).resolve().parent.parent
DEFAULT_CBF = PKG / "tests/data/example_1100.cbf.nc"
DEFAULT_CBR = PKG / "tests/data/assim_1100.cbr"


def starts_from_cbr(cbr, n):
    """Spread finite-P posterior draws from a CARDAMOM .cbr (netCDF)."""
    import netCDF4
    with netCDF4.Dataset(cbr) as ds:
        pars = np.array(ds["Parameters"][:])
    u = np.clip(np.asarray(par2nor(pars)), 1e-9, 1 - 1e-9)
    sel = np.linspace(0, len(u) - 1, n).astype(int)
    return np.log(u[sel] / (1 - u[sel]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cbf", default=str(DEFAULT_CBF))
    ap.add_argument("--starts", type=int, default=8)
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--screen", action="store_true",
                    help="find starts by blind prior rejection (GPU advised)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    print(f"[target] {args.cbf}", flush=True)
    logpost, _cbf = build_logpost(args.cbf)

    if args.screen:
        z0, P0, n_eval = find_feasible_starts(logpost, args.starts,
                                              seed=args.seed)
    else:
        z0 = starts_from_cbr(DEFAULT_CBR, args.starts)
        print(f"[starts] {len(z0)} spread posterior draws from bundled .cbr")

    t0 = time.time()
    res = multipoint_laplace(logpost, z0, max_iters=args.iters)
    print(f"[lbfgs] wall {time.time()-t0:.0f}s")

    keep = dedupe_modes(res["z_end"], res["P_end"])
    zM = res["z_end"][keep]
    jac = np.array([float(logit_jacobian(np.asarray(z))) for z in zM])
    print(f"[modes] {len(keep)} distinct; z-density P = "
          f"{np.round(res['P_end'][keep], 1)}")
    print(f"        model-P (Jacobian removed)   = "
          f"{np.round(res['P_end'][keep] - jac, 1)}")

    Hs = exact_hessians(logpost, zM)
    covs = np.array([cap_covariance(H) for H in Hs])
    wts = evidence_weights(res["P_end"][keep], covs)
    print(f"[laplace] evidence weights: {np.round(wts, 3)}")

    rng = np.random.default_rng(args.seed)
    zl = mixture_draws(rng, zM, covs, wts, 20000)
    u = 1 / (1 + np.exp(-zl))
    print(f"[laplace] u-space mixture sd (median over 89 params): "
          f"{np.median(u.std(axis=0)):.3f}")
    print("done. Modes -> chain seeds; covs[argmax(wts)] -> RWM proposal "
          "(see dalec_jax.inference.run_rwm); mixture -> screening only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
