"""Export a deterministic fixture + JAX reference outputs for the Julia port.

Everything is written as flat little-endian float32 (jax runs DifferLand in f32)
plus a JSON manifest carrying shapes, so the Julia side needs no NPZ dependency.

Reference values recorded per config: the loss, the full (T,32) output matrix,
and the gradient w.r.t. (param_initial, pool_initial, gpp_params).
"""
import json, os, sys
import numpy as np
import jax, jax.numpy as jnp

DL = "/export/data1/spandey/DifferLand/DifferLand_v1.1"
sys.path.insert(1, DL)
os.chdir(os.path.join(DL, "experiments"))

from DifferLand.model.DALEC990 import DALEC990
from DifferLand.util.init_mlp_params import init_mlp_params
from DifferLand.util.normalization import par2nor
from DifferLand.util.preprocessing import (get_train_test_sel, generate_met_matrix,
                                           generate_site_level_target_matrix)
from DifferLand.model.DALEC_990_parinfo import (dalec990_param_parmin, dalec990_pool_parmin,
                                                dalec990_parmin_arr, dalec990_parmax_arr)
import pandas as pd, xarray as xr

OUT = "/home/spandey/rough/cardamom_research/differland_jl/fixtures"
SITE = "US-Var"
K_ANNEAL = 30.0          # the value k is clamped to for ~most of a 25k run
# Calibrated operating points (best test-nNSE of the 40 shipped runs per config).
# A random init makes the annealed-k EDC penalty overflow float32 to +inf -- that is
# the real "numerical issue, reinitialize" path -- so it is useless as a fixture.
CONFIGS = {"default": 2, "nn_whole": 5}
CAL = {"default": "daily_US-Var_default_32_v6.pickle",
       "nn_whole": "daily_US-Var_nn_whole_1_v6.pickle"}

def w(name, arr):
    a = np.ascontiguousarray(np.asarray(arr, dtype=np.float32))
    a.tofile(os.path.join(OUT, name + ".bin"))
    return list(a.shape)

ds = xr.open_dataset(os.path.join(DL, "drivers", SITE + ".nc"))
train_sel, test_sel = get_train_test_sel(ds)
met = generate_met_matrix(ds, train_sel, test_sel, train_mode=False)
tgt = generate_site_level_target_matrix(ds, train_sel, train_mode=False, reco=True)
ce_opt = pd.read_csv("./ce_opt.csv").query("sitename == @SITE")["ce_opt"].values.item()
train_end_idx = int(np.sum(train_sel))

man = {"site": SITE, "T": int(met.shape[0]), "train_end_idx": train_end_idx,
       "k": K_ANNEAL, "ce_opt": float(ce_opt), "jax_version": jax.__version__,
       "shapes": {}, "configs": {}}
man["shapes"]["met"] = w("met", met)
man["shapes"]["target"] = w("target", tgt)
man["shapes"]["parmin_arr"] = w("parmin_arr", dalec990_parmin_arr)
man["shapes"]["parmax_arr"] = w("parmax_arr", dalec990_parmax_arr)
man["shapes"]["train_sel"] = w("train_sel", train_sel.astype(np.float32))
man["shapes"]["test_sel"] = w("test_sel", test_sel.astype(np.float32))

import pickle
CALDIR = "/export/data1/spandey/DifferLand/calibrated_parameters"

for name, cidx in CONFIGS.items():
    model = DALEC990(train_end_idx, water_stress_type=name, ce_opt=ce_opt, reco=True)
    with open(os.path.join(CALDIR, CAL[name]), "rb") as fh:
        cal = pickle.load(fh)
    param_initial, pool_initial, gpp_params = cal["param_state"]
    layers = ([int(np.asarray(gpp_params[0]["weights"]).shape[0])]
              + [int(np.asarray(l["weights"]).shape[1]) for l in gpp_params])
    ent = {"config_index": cidx, "layers": layers, "mlp": [],
           "calibrated_from": CAL[name],
           "nnse_eval": [float(x) for x in cal["nnse_eval"]],
           "param_initial": w(f"{name}_param_initial", param_initial),
           "pool_initial": w(f"{name}_pool_initial", pool_initial)}
    for li, layer in enumerate(gpp_params):
        ent["mlp"].append({
            "weights": w(f"{name}_W{li}", layer["weights"]),
            "biases":  w(f"{name}_b{li}", layer["biases"])})

    out = model.forward(param_initial, pool_initial, gpp_params, met)
    loss = model.compute_loss(param_initial, pool_initial, gpp_params, met, tgt, K_ANNEAL)
    g = jax.grad(model.compute_loss, argnums=[0, 1, 2])(
        param_initial, pool_initial, gpp_params, met, tgt, K_ANNEAL)

    ent["ref_loss"] = float(loss)
    ent["shapes"] = {"output": w(f"{name}_output", out),
                     "grad_param": w(f"{name}_grad_param", g[0]),
                     "grad_pool":  w(f"{name}_grad_pool",  g[1])}
    for li, layer in enumerate(g[2]):
        ent["shapes"][f"grad_W{li}"] = w(f"{name}_grad_W{li}", layer["weights"])
        ent["shapes"][f"grad_b{li}"] = w(f"{name}_grad_b{li}", layer["biases"])
    man["configs"][name] = ent
    print(f"{name}: loss={float(loss):.6f}  out={out.shape}  "
          f"|g_param|={float(jnp.linalg.norm(g[0])):.4e}  "
          f"grad_finite={bool(jnp.all(jnp.isfinite(g[0])) and jnp.all(jnp.isfinite(g[1])))}")

with open(os.path.join(OUT, "manifest.json"), "w") as fh:
    json.dump(man, fh, indent=2)
print("T =", met.shape[0], " train_end_idx =", train_end_idx, " ce_opt =", ce_opt)
