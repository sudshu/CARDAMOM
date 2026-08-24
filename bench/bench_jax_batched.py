"""The regime JAX is actually designed for: vmap over independent restarts.

Upstream runs 40 random restarts per (site, config) -- shipped
calibrated_parameters/ holds 16 sites x 6 configs x 40 = 3,840 calibrations.
Those are embarrassingly parallel, and batching them is where XLA-on-GPU should
win outright. This measures value+grad at batch B on one A100 and reports
per-member cost, so it can be put beside the 1-core Julia number directly.

Env: BATCHES=1,8,32,128,512  CUDA_VISIBLE_DEVICES=1
"""
import os, sys, time, json
import numpy as np
DL = "/export/data1/spandey/DifferLand/DifferLand_v1.1"
FIX = "/home/spandey/rough/cardamom_research/differland_jl/fixtures"
sys.path.insert(1, DL); os.chdir(os.path.join(DL, "experiments"))
import jax, jax.numpy as jnp, pickle, xarray as xr
from DifferLand.model.DALEC990 import DALEC990
from DifferLand.util.preprocessing import (get_train_test_sel, generate_met_matrix,
                                           generate_site_level_target_matrix)

BATCHES = [int(x) for x in os.environ.get("BATCHES", "1,8,32,128,512").split(",")]
man = json.load(open(os.path.join(FIX, "manifest.json")))
ds = xr.open_dataset(os.path.join(DL, "drivers", man["site"] + ".nc"))
tr, te = get_train_test_sel(ds)
met = generate_met_matrix(ds, tr, te, train_mode=False)
tgt = generate_site_level_target_matrix(ds, tr, train_mode=False, reco=True)
K = np.float32(man["k"])
res = {"device": str(jax.devices()[0]), "T": man["T"], "configs": {}}
print("device:", jax.devices(), flush=True)

for name, c in man["configs"].items():
    m = DALEC990(man["train_end_idx"], water_stress_type=name, ce_opt=man["ce_opt"], reco=True)
    with open(os.path.join("/export/data1/spandey/DifferLand/calibrated_parameters",
                           c["calibrated_from"]), "rb") as fh:
        p0, q0, gp = pickle.load(fh)["param_state"]
    f = lambda a, b, g: m.compute_loss(a, b, g, met, tgt, K)
    res["configs"][name] = {}
    for B in BATCHES:
        # perturb each member slightly so nothing can be CSE'd across the batch
        key = jax.random.PRNGKey(3)
        jit = lambda x: jnp.broadcast_to(x, (B,) + x.shape) * (
            1 + 1e-4 * jax.random.normal(key, (B,) + x.shape))
        P = jit(p0); Q = jit(q0)
        G = [{k2: jit(v) for k2, v in L.items()} for L in gp]
        vg = jax.jit(jax.vmap(jax.value_and_grad(f, argnums=[0, 1, 2])))
        t = time.perf_counter(); jax.block_until_ready(vg(P, Q, G))
        tcomp = time.perf_counter() - t
        best, n, t0 = float("inf"), 0, time.time()
        while n < 3 or time.time() - t0 < 3.0:
            t = time.perf_counter(); jax.block_until_ready(vg(P, Q, G))
            best = min(best, time.perf_counter() - t); n += 1
        res["configs"][name][B] = dict(total_ms=best*1e3, per_member_ms=best*1e3/B,
                                       compile_s=tcomp, reps=n)
        print(f"  {name:9s} B={B:4d}  {best*1e3:9.2f} ms total   "
              f"{best*1e3/B:8.4f} ms/member   (compile {tcomp:.1f} s, min of {n})", flush=True)
        json.dump(res, open("/home/spandey/rough/cardamom_research/differland_jl/results/jax_gpu_batched.json","w"), indent=2)
print("wrote batched results")
