# OSSE: scoring SARLA against a known truth at 89-D

A running record of an open experiment. Numbers here are measured; anything
not yet run is marked TBD.

## Why

Every comparison in [DEVLOG_OPTIMIZER.md](DEVLOG_OPTIMIZER.md) scores the
fast path (SARLA atlas + chart-shaped RWM) against the converged ADEMCMC
fleet. That is a proxy for truth, and on 2026-08-31 it was shown to be an
imperfect one: measured against the real observations, **ADEMCMC's own 90%
predictive bands hold only 31–65% of them**, so it is not calibrated either.
Treating it as ground truth has been flattering it.

The 24-D toy (`scripts/toy_mid.py`) does give exact ground truth, and it
earned its keep — it produced the chain-allocation fix and the
budget/coverage result, both of which transferred. But it also gets one
symptom backwards: the toy fails by *under*-covering well-separated basins,
while NL-Loo *over*-disperses across one dominant region. So the toy cannot
be used to chase the remaining defect.

An OSSE closes that gap: ground truth at the **actual 89-D problem, with the
actual model, the actual observation operators and the actual site drivers**.
"Which sampler is right" becomes answerable rather than assumed, and a
withheld-data test needs no extra assumptions because the truth over the
withheld period is known exactly.

## Precedent in the CARDAMOM literature

The 2025 CARDAMOM review (*Global Change Biology* 31:e70462) names this as
an active community direction:

> the CARDAMOM community has substantial opportunities to shape future
> satellite missions by conducting data assimilation experiments with
> existing (Smallman et al. 2017, 2021) and synthetic datasets (Holtzman et
> al. 2023) to refine observation requirements and quantify the potential
> value of observations.

The closest methodological precedent is **Famiglietti et al. 2021**,
"Optimal model complexity for terrestrial carbon cycle prediction"
(*Biogeosciences* 18:2727) — the COMPLEX experiment: 16 structurally distinct
DALEC variants calibrated and validated at 6 eddy-covariance sites. Two
design choices adopted here from that work:

1. **Calibration/validation split by time period at flux sites.**
2. **Forecast skill on withheld data as the headline metric**, not parameter
   recovery alone. Their conclusion — increased complexity only improves
   forecast skill if the parameters are adequately informed, and otherwise
   degrades it — is about the informativeness of the data, which is exactly
   what an OSSE controls.

The review also gives the community's own cost figure, useful context for
the speed claims on the other page:

> Each CARDAMOM location or pixel requires 10⁵–10⁹ model simulations to
> reach convergence.

Our ADEMCMC reference used 500,000 iterations — the bottom of that range.

## Design

`scripts/osse_make.py` (research repo).

**θ_true.** A *typical* ADEMCMC draw — the one whose log-posterior is
nearest the fleet median (−218.42, against a fleet best of −192.81), not the
best one. An atypical or mode-adjacent truth would flatter whichever sampler
chases modes. Being a real accepted draw, it is guaranteed EDC-feasible.

**Pseudo-observations.** θ_true is pushed through the likelihood's own
observation operators (`likelihood/__init__.py:189-204`) — not a
re-derivation — then corrupted with the noise model declared in the site's
own CBF attributes:

| stream | `opt_unc_type` | unc | noise generated |
| --- | --- | --- | --- |
| GPP | 1 (log-space) | `single_unc` 3.0 | lognormal, σ = ln 3 |
| ET | 1 (log-space) | `single_unc` 3.0 | lognormal, σ = ln 3 |
| NBE | 0 (additive) | `single_unc` 1.0 | N(0, 1) |
| LAI | 1 (log-space) | `single_annual_unc` 1.5, `opt_filter` 3 | lognormal, σ = ln 1.5 |

Verified rather than assumed: `opt_unc_type == 1` log-transforms both model
and observations before the residual, so the likelihood is Gaussian in log
space and multiplicative lognormal generation is **exactly** self-consistent.
The inference is therefore correctly specified — it is not fighting a
mismatched error model, which would confound a sampler comparison.

Note the consequence: a factor-3 1σ uncertainty is very weak data. Pseudo
GPP has mean 6.72 against a truth mean of 3.52 (lognormal mean inflation,
exp(σ²/2) = 1.83). This is what the likelihood believes the data to be, and
the resulting posterior really should be broad.

**Observation geometry, discovered while building this.** At NL-Loo the flux
observations occupy steps **0–83** and LAI steps **24–107**, of a 192-step
record. So:

- A first attempt that withheld "the last 25% of the record" withheld
  **nothing** and produced two identical CBFs. Fixed: withholding is now by
  observation *count* within each stream.
- **Steps 108–191 have no observations at all, even in the real setup.**
  Every fit on this page and the last has been extrapolating over that final
  7 years, and it has never been checked. In an OSSE the truth there is
  known, so that window is a free projection test.

**Two datasets written:**

| file | GPP | NBE | ET | LAI | purpose |
| --- | --- | --- | --- | --- | --- |
| `osse_full.cbf.nc` | 84 | 84 | 84 | 65 | parameter recovery |
| `osse_holdout.cbf.nc` | 42 | 42 | 42 | 32 | forecast skill |

`osse_truth.npz` stores θ_true, the noiseless truth series, the pseudo-obs,
and the assimilated index sets.

## What will be measured

Against θ_true and the noiseless truth series, for each sampler:

1. **Parameter recovery** — per-parameter z-score of θ_true within the
   marginal, and the rank of θ_true among posterior draws. Under a
   calibrated posterior the 89 ranks are uniform; systematic deviation
   diagnoses over- or under-confidence *without* needing a second sampler.
2. **Forecast skill on withheld observations** (steps of the second half),
   RMSE/bias/correlation against the noiseless truth, plus coverage of the
   predictive interval.
3. **Projection skill over steps 108–191**, the never-observed tail.
4. **Science-unit recovery** — the eight quantities from the
   [science table](DEVLOG_OPTIMIZER.md), now against truth rather than
   against ADEMCMC. This is the one that answers whether the residual
   sampler differences matter.

Both samplers are scored identically. The open question this settles: at
NL-Loo the fast path is *over*-dispersed relative to ADEMCMC by 3.5–6.3×,
yet its predictive intervals are *better* calibrated against the real data
(0.83 vs 0.65 coverage on NBE). Only a known truth can say which of those
two facts is the important one.

## Status

| step | state |
| --- | --- |
| pseudo-observations generated | **done** (`runs/osse/`) |
| fast path on `osse_full` | TBD |
| fast path on `osse_holdout` | TBD |
| ADEMCMC on `osse_full` | TBD (~33 h, 64 chains) |
| ADEMCMC on `osse_holdout` | TBD (~33 h, 64 chains) |
| scoring | TBD |

Cost note: the fast path is ~37 min per fit (8 min CPU atlas + 28 min GPU
sampling); ADEMCMC is ~33 h per fit. The two ADEMCMC arms are the schedule
constraint, and the host is shared — at the time of writing its load average
was 155 of 256 cores with a colleague's job on GPU1.

## Reproducing

```bash
# 1. pseudo-observations (GPU, ~2 min)
env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/osse_make.py

# 2. fast path: atlas charts on CPU (Hessians need host memory), then sample
env -u LD_LIBRARY_PATH JAX_PLATFORMS=cpu .venv/bin/python scripts/nlloo_build_charts.py
env -u LD_LIBRARY_PATH CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/nlloo_budget.py

# 3. ADEMCMC arms (background, ~33 h each)
NCHAINS=32 SRC_CBF=runs/osse/osse_full.cbf.nc bash scripts/ademcmc_fleet_nlloo.sh
```

`env -u LD_LIBRARY_PATH` is mandatory — the login shell's CUDA path shadows
the pip wheels and silently drops JAX to CPU.

## Context

- Sampler development and the results this is testing:
  [DEVLOG_OPTIMIZER.md](DEVLOG_OPTIMIZER.md)
- Toy target with exact ground truth: `scripts/toy_mid.py`
- Famiglietti et al. 2021, *Biogeosciences* 18:2727 —
  https://bg.copernicus.org/articles/18/2727/2021/
- CARDAMOM review, *Global Change Biology* 2025, 31:e70462 —
  https://doi.org/10.1111/gcb.70462
