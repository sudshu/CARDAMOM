"""Is the residual Julia-vs-JAX gap float32 chaos or a transcription bug?

Run the SAME jax code in float64 and measure how far jax-float32 sits from it.
If jax's own f32 error is the same size as julia's f32 error, the model is
float32-chaotic at that magnitude and the port is exonerated.
"""
import json, os, sys
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

DL = "/export/data1/spandey/DifferLand/DifferLand_v1.1"
sys.path.insert(1, DL); os.chdir(os.path.join(DL, "experiments"))
FIX = "/home/spandey/rough/cardamom_research/differland_jl/fixtures"

from DifferLand.model.DALEC990 import DALEC990
import DifferLand.model.DALEC_990_parinfo as PI
import pickle, pandas as pd, xarray as xr
from DifferLand.util.preprocessing import (get_train_test_sel, generate_met_matrix,
                                           generate_site_level_target_matrix)

man = json.load(open(os.path.join(FIX, "manifest.json")))
T, n, K = man["T"], man["train_end_idx"], man["k"]
rd = lambda nm, *sh: np.fromfile(os.path.join(FIX, nm + ".bin"), np.float32).reshape(*sh)

ds = xr.open_dataset(os.path.join(DL, "drivers", man["site"] + ".nc"))
tr, te = get_train_test_sel(ds)
met32 = np.asarray(generate_met_matrix(ds, tr, te, train_mode=False), np.float32)
tgt32 = np.asarray(generate_site_level_target_matrix(ds, tr, train_mode=False, reco=True), np.float32)

# monkeypatch the module-level float32 bound arrays up to float64
for nm in ("dalec990_parmin_arr", "dalec990_parmax_arr", "dalec990_param_parmin",
           "dalec990_param_parmax", "dalec990_pool_parmin", "dalec990_pool_parmax"):
    setattr(PI, nm, jnp.asarray(getattr(PI, nm), dtype=jnp.float64))
PI.dalec990_parmin = PI.DALEC990ParamBounds(*PI.dalec990_parmin_arr)
PI.dalec990_parmax = PI.DALEC990ParamBounds(*PI.dalec990_parmax_arr)
import importlib, DifferLand.model.DALEC990 as M990
importlib.reload(M990)
from DifferLand.model.DALEC990 import DALEC990 as D64

ce_opt = man["ce_opt"]
report = {}
for name, c in man["configs"].items():
    m = D64(n, water_stress_type=name, ce_opt=ce_opt, reco=True)
    p32 = rd(f"{name}_param_initial", 30); q32 = rd(f"{name}_pool_initial", 8)
    with open(os.path.join("/export/data1/spandey/DifferLand/calibrated_parameters",
                           c["calibrated_from"]), "rb") as fh:
        gp32 = pickle.load(fh)["param_state"][2]

    up = lambda a: jnp.asarray(np.asarray(a), dtype=jnp.float64)
    args64 = (up(p32), up(q32), [{k2: up(v) for k2, v in L.items()} for L in gp32],
              up(met64 := met32.astype(np.float64)), up(tgt32.astype(np.float64)), np.float64(K))
    out64 = np.asarray(m.forward(*args64[:4]), np.float64)
    L64 = float(m.compute_loss(*args64))
    g64 = jax.grad(m.compute_loss, argnums=[0, 1, 2])(*args64)
    g64f = np.concatenate([np.asarray(g64[0]).ravel(), np.asarray(g64[1]).ravel()] +
        [x for L in g64[2] for x in (np.asarray(L["weights"]).ravel(),
                                     np.asarray(L["biases"]).ravel())])

    out32 = rd(f"{name}_output", T, 32).astype(np.float64)
    g32f = np.concatenate([rd(f"{name}_grad_param", 30), rd(f"{name}_grad_pool", 8)] +
        [x for li in range(len(c["mlp"])) for x in
         (rd(f"{name}_grad_W{li}", *c["mlp"][li]["weights"]).ravel(),
          rd(f"{name}_grad_b{li}", *c["mlp"][li]["biases"]).ravel())])

    rms = np.sqrt((out64 ** 2).mean(axis=0))
    denom = np.maximum(np.abs(out64), np.maximum(rms, 1e-30))
    traj_err = float(np.max(np.abs(out32 - out64) / denom))
    gn = np.linalg.norm(g64f)
    report[name] = dict(
        loss_f64=L64, loss_f32=c["ref_loss"],
        loss_relerr=abs(c["ref_loss"] - L64) / abs(L64),
        traj_scaled_maxerr=traj_err,
        grad_relerr=float(np.linalg.norm(g32f - g64f) / gn),
        grad_cos=float(g32f @ g64f / (gn * np.linalg.norm(g32f))))
    np.asarray(out64, np.float64).tofile(os.path.join(FIX, f"{name}_output_f64.bin"))
    g64f.astype(np.float64).tofile(os.path.join(FIX, f"{name}_grad_f64.bin"))
    print(f"[{name}] jax-f32 vs jax-f64:  loss rel {report[name]['loss_relerr']:.3e}   "
          f"traj scaled max {traj_err:.3e}   grad rel {report[name]['grad_relerr']:.3e}   "
          f"cos {report[name]['grad_cos']:.10f}")

json.dump(report, open(os.path.join(FIX, "f64_reference.json"), "w"), indent=2)
