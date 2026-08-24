"""Fairness probe: does XLA's CPU backend get anything from extra cores on this
workload? A 3,287-step sequential scan cannot parallelize over time; only the
reductions could. Measures value+grad at several core counts."""
import os, sys, time, json
os.environ.setdefault("JAX_PLATFORMS", "cpu")
NC = int(os.environ.get("NCORES", "1"))
DL = "/export/data1/spandey/DifferLand/DifferLand_v1.1"
FIX = "/home/spandey/rough/cardamom_research/differland_jl/fixtures"
sys.path.insert(1, DL); os.chdir(os.path.join(DL, "experiments"))
import jax, numpy as np, pickle, xarray as xr
from DifferLand.model.DALEC990 import DALEC990
from DifferLand.util.preprocessing import (get_train_test_sel, generate_met_matrix,
                                           generate_site_level_target_matrix)
man = json.load(open(os.path.join(FIX, "manifest.json")))
ds = xr.open_dataset(os.path.join(DL, "drivers", man["site"] + ".nc"))
tr, te = get_train_test_sel(ds)
met = generate_met_matrix(ds, tr, te, train_mode=False)
tgt = generate_site_level_target_matrix(ds, tr, train_mode=False, reco=True)
K = np.float32(man["k"])
out = {}
for name, c in man["configs"].items():
    m = DALEC990(man["train_end_idx"], water_stress_type=name, ce_opt=man["ce_opt"], reco=True)
    with open(os.path.join("/export/data1/spandey/DifferLand/calibrated_parameters",
                           c["calibrated_from"]), "rb") as fh:
        p0, q0, gp = pickle.load(fh)["param_state"]
    vg = jax.jit(jax.value_and_grad(lambda a,b,g: m.compute_loss(a,b,g,met,tgt,K), argnums=[0,1,2]))
    jax.block_until_ready(vg(p0,q0,gp))
    best, t0, n = float("inf"), time.time(), 0
    while n < 5 or time.time()-t0 < 2.0:
        t = time.perf_counter(); jax.block_until_ready(vg(p0,q0,gp))
        best = min(best, time.perf_counter()-t); n += 1
    out[name] = best*1e3
    print(f"  {name:9s} value+grad {best*1e3:7.3f} ms  (cores={NC}, min of {n})", flush=True)
json.dump(out, open(f"/home/spandey/rough/cardamom_research/differland_jl/results/jax_cpu_cores_{NC}.json","w"))
