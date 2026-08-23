# BUG_COMPAT registry — C behaviours reproduced deliberately

The JAX port reproduces these C behaviours exactly (`reproduce=True` in
`src/dalec_jax/bug_compat.py`). None may be "fixed" without explicit user
approval and a registry update; every code site carries `# BUG_COMPAT: <id>`.

| id | C site | Behaviour | Consequence |
| --- | --- | --- | --- |
| `q_ly1_overflow` | DALEC_1100.c:784 | LY2 overflow accumulates into `FLUXES[F.q_ly1]` (probable copy-paste; LY1 block writes the same slot) | Per-layer runoff split q_ly1/q_ly2 wrong; ROFF total unaffected |
| `pi_7digit` | GLOBAL_CONSTANTS.c (DGCM_PI) | π = 3.1415927 (float32-precision literal in double code) | Solar geometry / daylength differ from full-precision π; port must use the 7-digit literal |
| `lf_exact_eq` | DALEC_1100.c:506 | `(D_LF_LY1 + D_LF_LY2) == 2` exact float equality gates hydraulic mortality | Works only because SOIL_TEMP assigns literal 1.0; float64 mandatory; do not weaken to >= |
| `isfinite_freeze` | DALEC_1100.c:1137-1141 | Non-finite prognostic pool at step n+1 → break; the poisoned step IS written; steps > n+1 remain calloc-zero | JAX carries `alive` flag; zeros (not NaN) after break; break index asserted exactly in tests |
| `edc_shortcircuit_stale` | DALEC_EDC_FUNCTIONS.c:43 | After first −inf EDC, later EDCs are not evaluated; their M_EDCs slots keep stale values | JAX evaluates all EDCs but reproduces gate arithmetic; comparisons mask the stale tail |
| `runmodel_stale_write` | DALEC_MLF2.c:47 + CARDAMOM_RUN_MODEL.c:424-435 | Model skipped for prerun-EDC-failing samples; RUN_MODEL writes the previous sample's trajectory under the new sample's PARS | RUN_MODEL output never used as oracle; `trajectory` subcommand calls DALEC_1100 directly |
| `obs_minmax_uninit` | CARDAMOM_LIKELIHOOD_FUNCTION.c:191-192 | `OBS.min_value/max_value` read while never initialized (MINMAX code commented out) when setting `validobs` | Latent UB in C; port checks what consumes `.validobs` and documents; no numeric effect observed via `values != DEFAULT_DOUBLE_VAL` path |
| `lst_dead_store` | DALEC_1100.c:53-54 | `LST` computed with int-truncation+modulo then immediately overwritten by `LST=0.5*24*60` | Port transcribes only line 54 (the live store) |
| `obsope_int_div_index` | DALEC_1100.c:1574 | `OBSOPE.rhch4_rhco2_flux = F.rh_ch4/F.rh_co2` — C INTEGER division of two flux indices, then used as a PARS index in DALEC_OBSOPE_rhch4_rhco2 (`1 - M_PARS[that]`) | PEQ_rhch4_rhco2 term reads a parameter chosen by index arithmetic, not the intended flux ratio; inert unless the CBF carries that obs |
| `obsope_c3frac_unset` | DALEC_1100.c:1558-1559 | `OBSOPE.C3frac_PARAM` assignment is commented out; static zero-init leaves it 0 | PEQ_C3frac reads pars[0] (i_SWE); inert unless the CBF carries that obs |
| `unused_unc_9999` | CARDAMOM_LIKELIHOOD_FUNCTION.c:87-104 | Streams with no unc info get unc backfilled to -9999 then quadratured to +9999 | Harmless for filter modes that never read unc[]; reproduced so unc arrays compare equal |

Related but out of scope (scientific change, needs approval): un-clamped
explicit-Euler decomposition (D1100:922-939) with uncapped fT
(HET_RESP_RATES_JCR.c:78) drives the EDC-7 viability bottleneck; a cap or
sub-stepping would alter the model. The port reproduces the divergence.
