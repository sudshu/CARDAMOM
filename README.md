# DifferLand.jl — Julia port of the differentiable DALEC990

A Julia transcription of the JAX hybrid land model in
[DifferLand v1.1](https://github.com/JianingFang/DifferLand_Global) (Fang &
Gentine), built to answer one question: **for this workload, is JAX the right
tool?**

**Upstream:** [JianingFang/DifferLand_Global](https://github.com/JianingFang/DifferLand_Global)
(DifferLand v1.1, MIT, © 2023 Jianing Fang). The JAX code is the source of truth
for behaviour; this port reproduces it, including its defects. On the machine
this was developed on, upstream lives at
`/export/data1/spandey/DifferLand/DifferLand_v1.1/DifferLand/`.

**Reference:** Fang & Gentine, "Exploring Optimal Complexity for Soil Water
Stress Representation: A Hybrid-Machine Learning Approach."

See [LICENSE](LICENSE) — the upstream MIT notice is preserved there.

## What is ported

| Upstream (JAX) | Here (Julia) |
| --- | --- |
| `model/DALEC990.py::step` | `src/step.jl` |
| `model/DALEC990.py::forward` (`lax.scan`) | `src/forward.jl` |
| `model/DALEC990.py::{pre_edc, post_edc, compute_loss}` | `src/loss.jl` |
| `model/auxi/ACM.py` | `src/acm.jl` |
| `model/auxi/phenology.py` | `src/phenology.jl` |
| `model/DALEC_990_parinfo.py` | `src/parinfo.jl` |
| `util/normalization.py` | `src/normalization.jl` |
| `optimization/forward.py` (MLP) | `src/mlp.jl` |
| `optimization/loss_functions.py::{compute_nnse, negative_log_sigmoid}` | `src/loss.jl` |
| `optax.adam` (as used by `experiments/calibration.py`) | `src/adam.jl` |

Two of the six water-stress configurations, chosen to bracket the compute
profile: **config 2 (`default`, β-JS)** — pure physics, transcendental-heavy —
and **config 5 (`nn_whole`)** — GPP and ET both from an MLP, so BLAS-bound.
`float32` throughout, matching upstream (which never enables x64).

## Transcription rules

1. **Statement order is preserved.** Several statements in the pool-clamp
   cascade overwrite a flux that a later statement reads; reordering changes
   results. No algebraic simplification — `(1 - (1-r)^Δt)/Δt` stays as written,
   because `1-(1-r) ≠ r` in floating point.
2. `jnp.where` → `ifelse`, never a branch. Both arms are evaluated in both
   languages, so NaN propagation and the reverse-mode rule match.
3. **Upstream defects are reproduced, not fixed.** Each site is marked
   `BUGCOMPAT <id>` in the code:
   - `cfol_min_leaf_litter` (`DALEC990.py:163`) — the `Cfol` lower clamp sets
     `leaf_litter = (1-Cfol_min_sel)*parmin.Cfol`, which for a scalar bool is
     just `parmin.Cfol`, not the flux-balance expression every sibling clamp
     uses.
   - `puw_max_uses_min_sel` (`DALEC990.py:212`) — the PUW *upper* clamp gates
     its `q_puw` correction on `next_puw_min_sel`.
   - `acm_dayl_zero` (`ACM.py:38`) — the final day-length line adds
     `(1-mult_geq_one_sel)*0`, so it only masks by `mult > -1`.
   - `pre_edc` is handed `mean(met[:, 13])`, the mean of the *normalized*
     temperature (≈ 0 by construction), not the mean temperature.
4. Bounds and output-column indices come from `src/parinfo.jl`, transcribed
   from `DALEC_990_parinfo.py`. No hand-written index literals elsewhere.

## Deliberate implementation differences (not behaviour changes)

- Met and target matrices are stored **transposed**, `(18,T)` and `(10,T)`, so a
  timestep is one contiguous column. XLA assigns its own layout on the JAX side,
  so this is a layout choice, not an advantage taken.
- The forward pass writes its `(32,T)` output into a **caller-owned buffer**
  instead of stacking per-step results. This is the one structural difference
  that JAX cannot express, and it is part of what is being measured.
- MLP weights become `SMatrix` once per loss call, then live in registers
  through the scan.

## Verification

`test/verify.jl`. Float32 agreement alone would be a weak bar — the model is
float32-chaotic over 3,287 sequential steps — so the gate is *relative*:

| | jl-f32 vs jax-f32 | jl-f32 vs f64 | jax-f32 vs f64 |
| --- | ---: | ---: | ---: |
| loss, config 2 | 7.4e-08 | 7.6e-06 | 7.6e-06 |
| trajectory (32×3287, scaled max) | 2.7e-04 | 5.408e-03 | 5.408e-03 |
| gradient (L2 rel, 42 params) | 1.3e-04 | 8.98e-04 | 8.58e-04 |
| loss, config 5 | 1.4e-07 | 2.3e-06 | 2.4e-06 |
| trajectory | 1.0e-04 | 5.209e-03 | 5.208e-03 |
| gradient (240 params) | 1.1e-04 | 8.84e-04 | 7.86e-04 |

Julia agrees with JAX **20–50× more tightly than JAX agrees with its own
float64 answer**, and Julia's distance from the float64 truth is
indistinguishable from JAX's. Gradient cosine vs JAX: 0.9999999918 (config 2),
0.9999999958 (config 5).

End-to-end: 2,000 real Adam iterations from the same calibrated start agree to
~1e-7 for the first 20 iterations and have drifted only ~5e-5 by iteration
1,500 — the two calibrations are the same optimization.

The float64 reference comes from `bench/f64_reference.py`, which re-runs the
*upstream JAX code* with `jax_enable_x64` and its bound arrays upcast.

## Layout

```
src/            the port
test/verify.jl  the equivalence gate
bench/          export_fixture.py, f64_reference.py, bench_julia.jl, bench_jax.py
fixtures/       deterministic inputs + JAX reference outputs (float32 and float64)
results/        benchmark JSON
```

## Running

```bash
# fixtures (needs the DifferLand conda env)
cd /export/data1/spandey/DifferLand && env -u LD_LIBRARY_PATH \
  ./env/bin/python3 <this>/bench/export_fixture.py
env -u LD_LIBRARY_PATH JAX_PLATFORMS=cpu ./env/bin/python3 <this>/bench/f64_reference.py

# verify + benchmark
julia --project=. test/verify.jl
taskset -c 200 julia -t 1 --project=. bench/bench_julia.jl
taskset -c 201 env -u LD_LIBRARY_PATH DEV=cpu JAX_PLATFORMS=cpu ./env/bin/python3 bench/bench_jax.py
env -u LD_LIBRARY_PATH DEV=gpu JAX_PLATFORMS=cuda CUDA_VISIBLE_DEVICES=1 ./env/bin/python3 bench/bench_jax.py
```

`env -u LD_LIBRARY_PATH` matters: the login shell's `/usr/local/cuda/lib64`
shadows the pip CUDA wheels and silently drops JAX to CPU.
