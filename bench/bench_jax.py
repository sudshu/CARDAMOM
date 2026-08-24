"""JAX-side performance for DifferLand DALEC990, matched to bench_julia.jl.

Same site (US-Var, T=3287), same float32, same calibrated starting point, same
k. Every timing blocks on the device. JIT compile time is measured on the first
call and reported separately, never folded into steady state.

Two end-to-end numbers are reported because they are genuinely different things:
  as-shipped  -- the upstream loop, which reads `loss` every iteration to test
                 for NaN, forcing a device sync (this is what calibration.py does)
  async       -- the same loop with the sync moved to the end (best case for JAX)
Julia's loop reads the loss every iteration too, so `as-shipped` is the
semantically matched comparison; `async` is JAX's ceiling.

Env: DEV=cpu|gpu   CALIB_ITERS=n   OUTJSON=path
"""
import json, os, sys, time
import numpy as np

DEV = os.environ.get("DEV", "cpu")
CALIB_ITERS = int(os.environ.get("CALIB_ITERS", "2000"))
OUT = os.environ.get("OUTJSON", "/home/spandey/rough/cardamom_research/differland_jl/results/jax_%s.json" % DEV)
os.environ.setdefault("JAX_PLATFORMS", "cpu" if DEV == "cpu" else "cuda")

DL = "/export/data1/spandey/DifferLand/DifferLand_v1.1"
FIX = "/home/spandey/rough/cardamom_research/differland_jl/fixtures"
sys.path.insert(1, DL); os.chdir(os.path.join(DL, "experiments"))

import jax, jax.numpy as jnp, optax, pickle
from functools import partial
from DifferLand.model.DALEC990 import DALEC990
from DifferLand.util.preprocessing import (get_train_test_sel, generate_met_matrix,
                                           generate_site_level_target_matrix)
import xarray as xr

man = json.load(open(os.path.join(FIX, "manifest.json")))
T, N_TRAIN, K = man["T"], man["train_end_idx"], np.float32(man["k"])
ds = xr.open_dataset(os.path.join(DL, "drivers", man["site"] + ".nc"))
tr, te = get_train_test_sel(ds)
met = generate_met_matrix(ds, tr, te, train_mode=False)
tgt = generate_site_level_target_matrix(ds, tr, train_mode=False, reco=True)
CAL = "/export/data1/spandey/DifferLand/calibrated_parameters"

def timeit(fn, budget=float(os.environ.get("BUDGET", "3.0")), minreps=5):
    fn()
    best, n, t0 = float("inf"), 0, time.time()
    while n < minreps or time.time() - t0 < budget:
        t = time.perf_counter(); fn(); dt = time.perf_counter() - t
        best = min(best, dt); n += 1
        if n > 100000: break
    return best, n

res = {"device": DEV, "jax": jax.__version__, "devices": [str(d) for d in jax.devices()],
       "T": T, "configs": {}}
print("jax", jax.__version__, "devices:", jax.devices(), " T =", T, flush=True)

for name, c in man["configs"].items():
    model = DALEC990(N_TRAIN, water_stress_type=name, ce_opt=man["ce_opt"], reco=True)
    with open(os.path.join(CAL, c["calibrated_from"]), "rb") as fh:
        p0, q0, gp = pickle.load(fh)["param_state"]
    nparams = int(sum(x.size for x in jax.tree_util.tree_leaves((p0, q0, gp))))

    fwd = jax.jit(lambda a, b, g: model.forward(a, b, g, met))
    lss = jax.jit(lambda a, b, g: model.compute_loss(a, b, g, met, tgt, K))
    vg  = jax.jit(jax.value_and_grad(
        lambda a, b, g: model.compute_loss(a, b, g, met, tgt, K), argnums=[0, 1, 2]))

    tc = {}
    t = time.perf_counter(); jax.block_until_ready(fwd(p0, q0, gp));  tc["forward"] = time.perf_counter() - t
    t = time.perf_counter(); jax.block_until_ready(lss(p0, q0, gp));  tc["loss"]    = time.perf_counter() - t
    t = time.perf_counter(); jax.block_until_ready(vg(p0, q0, gp));   tc["grad"]    = time.perf_counter() - t

    t_fwd,  n1 = timeit(lambda: jax.block_until_ready(fwd(p0, q0, gp)))
    t_loss, n2 = timeit(lambda: jax.block_until_ready(lss(p0, q0, gp)))
    t_grad, n3 = timeit(lambda: jax.block_until_ready(vg(p0, q0, gp)))

    # ---- end-to-end Adam, exactly the upstream update() ----
    tx = optax.adam(learning_rate=5e-4)
    lgf = jax.value_and_grad(partial(model.compute_loss), [0, 1, 2])

    @jax.jit
    def update(params, opt_state, k):
        a, b, g = params
        loss, grads = lgf(a, b, g, met, tgt, k)
        upd, opt_state = tx.update(grads, opt_state)
        return loss, optax.apply_updates(params, upd), opt_state

    def run_calib(sync_each_iter, iters):
        ps = (p0, q0, gp); st = tx.init(ps); trace = []
        loss, ps, st = update(ps, st, K)            # compile + warm
        ps = (p0, q0, gp); st = tx.init(ps)
        t0 = time.perf_counter()
        for i in range(iters):
            loss, ps, st = update(ps, st, K)
            if sync_each_iter:
                fl = float(loss)                    # the NaN test upstream performs
                if i < 20 or i % 500 == 0: trace.append(fl)
                if not np.isfinite(fl): break
        jax.block_until_ready((loss, ps))
        return time.perf_counter() - t0, trace

    t_sync, trace = run_calib(True,  CALIB_ITERS)
    t_async, _    = run_calib(False, CALIB_ITERS)

    print(f"--- {name} ({nparams} params) ---")
    print(f"  forward only            {t_fwd*1e3:8.3f} ms   (min of {n1})")
    print(f"  loss (fwd + reductions) {t_loss*1e3:8.3f} ms   (min of {n2})")
    print(f"  value+grad (jax.grad)   {t_grad*1e3:8.3f} ms   (min of {n3})   "
          f"overhead vs loss {t_grad/t_loss:.2f}x")
    print(f"  Adam iter, as-shipped   {t_sync/CALIB_ITERS*1e3:8.3f} ms   "
          f"-> 25k calib {t_sync/CALIB_ITERS*25000:.1f} s")
    print(f"  Adam iter, async        {t_async/CALIB_ITERS*1e3:8.3f} ms   "
          f"-> 25k calib {t_async/CALIB_ITERS*25000:.1f} s")
    print(f"  JIT compile: fwd {tc['forward']:.2f} s  loss {tc['loss']:.2f} s  grad {tc['grad']:.2f} s")
    if trace: print(f"  loss after {CALIB_ITERS} iters: {trace[-1]:.6f} (started {trace[0]:.6f})")
    print(flush=True)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    res["configs"][name] = dict(
        nparams=nparams, forward_ms=t_fwd*1e3, loss_ms=t_loss*1e3, grad_ms=t_grad*1e3,
        grad_over_loss=t_grad/t_loss,
        adam_iter_ms_sync=t_sync/CALIB_ITERS*1e3, adam_iter_ms_async=t_async/CALIB_ITERS*1e3,
        calib_25k_s_sync=t_sync/CALIB_ITERS*25000, calib_25k_s_async=t_async/CALIB_ITERS*25000,
        compile_s=tc, calib_iters=CALIB_ITERS, loss_trace=trace)
    json.dump(res, open(OUT, "w"), indent=2)     # flush after every config

print("wrote", OUT)
