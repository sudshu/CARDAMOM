"""DALEC_1100 forward model — literal transcription of DALEC_1100.c:246-1148.

Structure mirrors the C exactly:
  - prederive_vegk: PREDERIVE_DALEC_1100_DATA (numpy — bit-identical libm)
  - _initial_pools: the init block (D1100.c:297-407)
  - _step: one timestep (D1100.c:459-1141), including the FOUR sequential
    carbon-removal passes and the order-critical q_ly1/q_ly3 overflow
    mutations (BUG_COMPAT: q_ly1_overflow at :784)
  - run_dalec_1100: lax.scan with the alive-flag freeze reproducing the
    isfinite break (BUG_COMPAT: isfinite_freeze; poisoned step recorded,
    zeros after — per-sample-zeroed-buffer canonical form)

Transcription rules per CLAUDE.md: op order preserved, jnp.fmax/fmin for C
fmax/fmin, both-branch-safe wheres only where they cannot change the selected
value, constants/indices from constants.py / generated indices.py only.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp
from jax import lax

from ..constants import (DGCM_LATENT_HEAT_FUSION_3,
                         DGCM_LATENT_HEAT_VAPORIZATION, DGCM_PI, DGCM_SEC_DAY,
                         DGCM_TK0C)
from ..indices import F, NOFLUXES, NOPOOLS, P, S
from ..modules.alloc_and_auto_resp import alloc_and_auto_resp_fluxes
from ..modules.drainage import drainage
from ..modules.het_resp_rates_jcr import het_resp_rates_jcr
from ..modules.hydrofun import (hydrofun_ewt2moi, hydrofun_moi2con,
                                hydrofun_moi2ewt, hydrofun_moi2psi)
from ..modules.knorr_allocation import knorr_allocation
from ..modules.liu_an_et import liu_an_et
from ..modules.soil_energy import (initialize_internal_soil_energy,
                                   internal_energy_per_liquid_h2o_unit_mass)
from ..modules.soil_temp_liquid_frac import soil_temp_and_liquid_frac

PSI_POROSITY = -0.117 / 100          # D1100.c:330
MINPSI = -30.0                       # D1100.c:343
PREDERIVED_GEO_FLUX = 0.105 * 3600 * 24   # D1100.c:281
N_PROGNOSTIC = 14                    # isfinite check covers pools 0..13
LOG_DBL_MAX = 709.782712893384       # exp overflows to inf strictly above


def _sigmoid_lf(u, LF):
    """C pattern 1/(1+exp(u))*LF with a bit-exact overflow guard: for
    u > log(DBL_MAX) the C's exp is inf and the quotient +0.0; below the
    cutoff the operand passes through fmin unchanged. Keeps grad NaN-free."""
    # jnp.minimum (NOT fmin): NaN operands must propagate exactly as in C
    s = 1 / (1 + jnp.exp(jnp.minimum(u, LOG_DBL_MAX)))
    return jnp.where(u > LOG_DBL_MAX, 0.0, s) * LF


def _psi_clamped(sm, retention):
    """C pattern fmax(HYDROFUN_MOI2PSI(sm, psi_porosity, b), minpsi).
    For sm <= 0 the C's pow produces ±inf/NaN and fmax selects MINPSI; we
    select MINPSI directly (identical value) with a safe pow operand so the
    backward pass stays finite."""
    psi = hydrofun_moi2psi(jnp.where(sm > 0, sm, 1.0), PSI_POROSITY, retention)
    return jnp.where(sm > 0, jnp.fmax(psi, MINPSI), MINPSI)

# forcing column order for the scan xs matrix
MET_COLUMNS = ("SSRD", "T2M_MIN", "T2M_MAX", "CO2", "DOY", "TOTAL_PREC",
               "VPD", "BURNED_AREA", "SNOWFALL", "SKT", "STRD",
               "DISTURBANCE_FLUX", "YIELD")


def prederive_vegk(DOY: np.ndarray, LAT: float) -> np.ndarray:
    """PREDERIVE_DALEC_1100_DATA (D1100.c:29-78), in numpy (libm-exact)."""
    pi = DGCM_PI
    DOY = np.asarray(DOY, dtype=np.float64)
    B = (DOY - 81) * 2 * pi / 365.
    ET1 = 9.87 * np.sin(2 * B) - 7.53 * np.cos(B) - 1.5 * np.sin(B)
    DA = 23.45 * np.sin((284 + DOY) * 2 * pi / 365)
    # C computes an int-cast LST then immediately overwrites it
    # (BUG_COMPAT: lst_dead_store) — only the live store is ported.
    LST = 0.5 * 24 * 60
    AST = LST + ET1
    h = (AST - 12 * 60) / 4
    alpha = np.arcsin(
        np.sin(pi / 180 * LAT) * np.sin(pi / 180 * DA)
        + np.cos(pi / 180 * LAT) * np.cos(pi / 180. * DA) * np.cos(pi / 180 * h)
    ) * 180 / pi
    zenith_angle = np.fmin(89, 90 - alpha)
    LAD = 0.5
    return LAD / np.cos(zenith_angle / 180 * pi)


def _initial_pools(pars):
    """D1100.c:297-407 — returns POOLS[0,:] (30,)."""
    v = [None] * NOPOOLS
    v[S.C_lab] = pars[P.i_labile]
    v[S.C_fol] = pars[P.i_foliar]
    v[S.C_roo] = pars[P.i_root]
    v[S.C_woo] = pars[P.i_wood]
    v[S.C_cwd] = pars[P.i_cwd]
    v[S.C_lit] = pars[P.i_lit]
    v[S.C_som] = pars[P.i_som]
    v[S.H2O_LY1] = hydrofun_moi2ewt(pars[P.i_LY1_SM], pars[P.LY1_por], pars[P.LY1_z])
    v[S.H2O_LY2] = hydrofun_moi2ewt(pars[P.i_LY2_SM], pars[P.LY2_por], pars[P.LY2_z])
    v[S.H2O_LY3] = hydrofun_moi2ewt(pars[P.i_LY3_SM], pars[P.LY3_por], pars[P.LY3_z])
    v[S.H2O_SWE] = pars[P.i_SWE]
    v[S.E_LY1] = initialize_internal_soil_energy(
        pars[P.i_LY1_E], v[S.H2O_LY1], pars[P.LY1_vhc], pars[P.LY1_z])
    v[S.E_LY2] = initialize_internal_soil_energy(
        pars[P.i_LY2_E], v[S.H2O_LY2], pars[P.LY2_vhc], pars[P.LY2_z])
    v[S.E_LY3] = initialize_internal_soil_energy(
        pars[P.i_LY3_E], v[S.H2O_LY3], pars[P.LY3_vhc], pars[P.LY3_z])

    v[S.D_LAI] = v[S.C_fol] / pars[P.LCMA]
    v[S.D_SCF] = jnp.where(v[S.H2O_SWE] > 0,
                           v[S.H2O_SWE] / (v[S.H2O_SWE] + pars[P.scf_scalar]),
                           0.0)
    v[S.D_SM_LY1] = hydrofun_ewt2moi(v[S.H2O_LY1], pars[P.LY1_por], pars[P.LY1_z])
    v[S.D_SM_LY2] = hydrofun_ewt2moi(v[S.H2O_LY2], pars[P.LY2_por], pars[P.LY2_z])
    v[S.D_SM_LY3] = hydrofun_ewt2moi(v[S.H2O_LY3], pars[P.LY3_por], pars[P.LY3_z])
    v[S.D_PSI_LY1] = _psi_clamped(v[S.D_SM_LY1], pars[P.retention])
    v[S.D_PSI_LY2] = _psi_clamped(v[S.D_SM_LY2], pars[P.retention])
    v[S.D_PSI_LY3] = _psi_clamped(v[S.D_SM_LY3], pars[P.retention])

    T1, LF1 = soil_temp_and_liquid_frac(pars[P.LY1_vhc], pars[P.LY1_z],
                                        v[S.H2O_LY1], v[S.E_LY1])
    T2, LF2 = soil_temp_and_liquid_frac(pars[P.LY2_vhc], pars[P.LY2_z],
                                        v[S.H2O_LY2], v[S.E_LY2])
    T3, LF3 = soil_temp_and_liquid_frac(pars[P.LY3_vhc], pars[P.LY3_z],
                                        v[S.H2O_LY3], v[S.E_LY3])
    v[S.D_TEMP_LY1], v[S.D_LF_LY1] = T1, LF1
    v[S.D_TEMP_LY2], v[S.D_LF_LY2] = T2, LF2
    v[S.D_TEMP_LY3], v[S.D_LF_LY3] = T3, LF3

    v[S.M_LAI_TEMP] = pars[P.init_T_mem]
    v[S.M_LAI_MAX] = pars[P.init_LAIW_mem] * pars[P.lambda_max]

    assert all(x is not None for x in v)
    return jnp.stack([jnp.asarray(x, dtype=jnp.float64) for x in v])


def _step(pars, deltat, LAT, prev, met_row, VegK_n):
    """One timestep (D1100.c:459-1141). prev = POOLS[p,:]; returns
    (new_pools (30,), fluxes (100,))."""
    one_over_deltat = 1 / deltat
    (SSRD, T2M_MIN, T2M_MAX, CO2, DOY, PREC, VPD, BURNED_AREA, SNOWFALL,
     SKT, STRD, DIST, YIELD) = met_row

    LY1max = pars[P.LY1_por] * pars[P.LY1_z] * 1000
    LY2max = pars[P.LY2_por] * pars[P.LY2_z] * 1000
    LY3max = pars[P.LY3_por] * pars[P.LY3_z] * 1000

    fx = {}   # flux index -> value; scattered into the (100,) array at the end
    LAI = prev[S.D_LAI]

    # ---------------- cold-temperature stress factor (476-488)
    Tminmin = pars[P.Tminmin] - DGCM_TK0C
    Tminmax = pars[P.Tminmax] - DGCM_TK0C
    g = jnp.where(T2M_MIN < Tminmin, 0.0,
                  jnp.where(T2M_MIN > Tminmax, 1.0,
                            (T2M_MIN - Tminmin) / (Tminmax - Tminmin)))

    # ---------------- water stress (490-515)
    beta1 = _sigmoid_lf(pars[P.beta_lgr]
                        * (-1 * prev[S.D_PSI_LY1] / pars[P.psi_50] - 1),
                        prev[S.D_LF_LY1])
    beta2 = _sigmoid_lf(pars[P.beta_lgr]
                        * (-1 * prev[S.D_PSI_LY2] / pars[P.psi_50] - 1),
                        prev[S.D_LF_LY2])
    beta = (beta1 * pars[P.LY1_z] + beta2 * pars[P.LY2_z] * pars[P.root_frac]) \
        / (pars[P.LY1_z] + pars[P.LY2_z] * pars[P.root_frac])

    betaHMF_1 = _sigmoid_lf(pars[P.beta_lgrHMF]
                            * (-1 * prev[S.D_PSI_LY1] / pars[P.psi_50HMF] - 1),
                            prev[S.D_LF_LY1])
    betaHMF_2 = _sigmoid_lf(pars[P.beta_lgrHMF]
                            * (-1 * prev[S.D_PSI_LY2] / pars[P.psi_50HMF] - 1),
                            prev[S.D_LF_LY2])
    betaHMF = (betaHMF_1 * pars[P.LY1_z]
               + betaHMF_2 * pars[P.LY2_z] * pars[P.root_frac]) \
        / (pars[P.LY1_z] + pars[P.LY2_z] * pars[P.root_frac])

    # exact float equality on purpose (BUG_COMPAT: lf_exact_eq, D1100.c:506)
    HMF = jnp.where((prev[S.D_LF_LY1] + prev[S.D_LF_LY2]) == 2,
                    1 - betaHMF, 0.0)

    fx[F.beta_factor] = jnp.fmin(beta, g)
    fx[F.soil_beta_factor] = beta
    fx[F.hydraulic_mortality_factor] = HMF

    air_temp_k = DGCM_TK0C + 0.5 * (T2M_MIN + T2M_MAX)

    # ---------------- LIU photosynthesis/ET (522-577)
    (liu_An, liu_Ag, liu_Rd, liu_transp, liu_evap,
     LEAF_MORTALITY_FACTOR) = liu_an_et(
        SSRD * 1e6 / DGCM_SEC_DAY, VPD / 10, air_temp_k, pars[P.Vcmax25],
        CO2, fx[F.beta_factor], pars[P.Med_g1], LAI, pars[P.ga], VegK_n,
        pars[P.Tupp], pars[P.Tdown], 1., pars[P.clumping],
        pars[P.leaf_refl_par], pars[P.leaf_refl_nir], pars[P.maxPevap],
        PREC, pars[P.q10canopy], pars[P.rauto_mrd_q10], pars[P.canopyRdsf],
        prev[S.C_lab], deltat)

    fx[F.leaf_mortality_factor] = LEAF_MORTALITY_FACTOR
    fx[F.gpp] = liu_Ag
    fx[F.Rd] = liu_Rd
    fx[F.gppnet] = liu_An
    transp = liu_transp

    transp_split = jnp.logical_or(beta1 > 0, beta2 > 0)
    _tden = beta1 * pars[P.LY1_z] + beta2 * pars[P.LY2_z] * pars[P.root_frac]
    transp1_active = transp * beta1 * pars[P.LY1_z] \
        / jnp.where(transp_split, _tden, 1.0)
    fx[F.transp1] = jnp.where(transp_split, transp1_active, 0.0)
    fx[F.transp2] = jnp.where(transp_split, transp - fx[F.transp1], 0.0)
    fx[F.evap] = liu_evap

    # ---------------- snow (579-600)
    fx[F.snowfall] = SNOWFALL
    swe_1 = prev[S.H2O_SWE] + fx[F.snowfall] * deltat
    SCFtemp = swe_1 / (swe_1 + pars[P.scf_scalar])
    SNOWMELT = jnp.fmin(jnp.fmax((DGCM_TK0C + SKT - pars[P.min_melt])
                                 * pars[P.melt_slope], 0.0), 1.0) \
        * swe_1 * one_over_deltat
    SUBLIMATION = pars[P.sublimation_rate] * SSRD * SCFtemp
    # (hardened: swe_1 == 0 gives SNOWMELT = SUBLIMATION = 0 and the C's
    # slf = 0/0 = NaN selects the no-rescale branch; safe operands keep the
    # selected values bit-identical and the backward pass finite)
    slf = (SNOWMELT + SUBLIMATION) * deltat / jnp.where(swe_1 > 0, swe_1, 1.0)
    rescale = jnp.logical_and(swe_1 > 0, slf > 1)
    _slf_safe = jnp.where(rescale, slf, 1.0)
    fx[F.melt] = jnp.where(rescale, SNOWMELT / _slf_safe, SNOWMELT)
    fx[F.sublimation] = jnp.where(rescale, SUBLIMATION / _slf_safe, SUBLIMATION)
    swe_new = jnp.fmax(swe_1 - (fx[F.melt] + fx[F.sublimation]) * deltat, 0.0)
    fx[F.ets] = fx[F.evap] + fx[F.transp1] + fx[F.transp2] + fx[F.sublimation]

    # ---------------- energy balance (602-680)
    SWin = SSRD * 1e6 / DGCM_SEC_DAY
    SWout_snowfree = SWin * 0.5 * (pars[P.leaf_refl_par] + pars[P.leaf_refl_nir])
    snow_albedo = 0.9
    SWout = (1. - prev[S.D_SCF]) * SWout_snowfree \
        + prev[S.D_SCF] * (SWin * snow_albedo)
    sigma = 5.67 * 1e-8
    LWin = STRD * 1e6 / DGCM_SEC_DAY
    tskin_k = SKT + DGCM_TK0C
    LWout = sigma * (tskin_k * tskin_k) * (tskin_k * tskin_k)
    Rn = SWin - SWout + LWin - LWout
    fx[F.net_radiation] = Rn
    fx[F.SWin] = SWin
    fx[F.LWin] = LWin
    fx[F.SWout] = SWout
    fx[F.LWout] = LWout

    lambda_liquid = DGCM_LATENT_HEAT_VAPORIZATION
    lambda_solid = DGCM_LATENT_HEAT_FUSION_3 + DGCM_LATENT_HEAT_VAPORIZATION
    From_Liquid = fx[F.evap] + fx[F.transp1] + fx[F.transp2]
    From_Solid = fx[F.sublimation]
    LE = (lambda_liquid * From_Liquid + lambda_solid * From_Solid) / DGCM_SEC_DAY
    fx[F.latent_heat] = LE

    fx[F.ground_heat] = (pars[P.thermal_cond_surf]
                         * (tskin_k - prev[S.D_TEMP_LY1])
                         / (pars[P.LY1_z] * 0.5)) * (1. - prev[S.D_SCF])
    fx[F.gh_in] = fx[F.ground_heat] * DGCM_SEC_DAY
    fx[F.sensible_heat] = Rn - fx[F.ground_heat] - fx[F.latent_heat]

    # ---------------- infiltration + drainage (682-707)
    liquid_in = PREC - SNOWFALL + fx[F.melt]
    fx[F.infil] = pars[P.max_infil] * (1 - jnp.exp(-liquid_in / pars[P.max_infil]))
    fx[F.q_surf] = liquid_in - fx[F.infil]

    drain_LY1 = prev[S.D_LF_LY1] * drainage(
        prev[S.D_SM_LY1], pars[P.Q_excess], -pars[P.field_cap], PSI_POROSITY,
        pars[P.retention])
    drain_LY2 = prev[S.D_LF_LY2] * drainage(
        prev[S.D_SM_LY2], pars[P.Q_excess], -pars[P.field_cap], PSI_POROSITY,
        pars[P.retention])
    drain_LY3 = prev[S.D_LF_LY3] * drainage(
        prev[S.D_SM_LY3], pars[P.Q_excess], -pars[P.field_cap], PSI_POROSITY,
        pars[P.retention])
    q_ly1 = hydrofun_moi2ewt(drain_LY1, pars[P.LY1_por], pars[P.LY1_z]) * one_over_deltat
    q_ly2 = hydrofun_moi2ewt(drain_LY2, pars[P.LY2_por], pars[P.LY2_z]) * one_over_deltat
    q_ly3 = hydrofun_moi2ewt(drain_LY3, pars[P.LY3_por], pars[P.LY3_z]) * one_over_deltat

    k_LY1 = hydrofun_moi2con(prev[S.D_SM_LY1], pars[P.hydr_cond], pars[P.retention])
    k_LY2 = hydrofun_moi2con(prev[S.D_SM_LY2], pars[P.hydr_cond], pars[P.retention])
    k_LY3 = hydrofun_moi2con(prev[S.D_SM_LY3], pars[P.hydr_cond], pars[P.retention])

    # ---------------- LY1<->LY2 transfer (709-736); q_* still pre-overflow
    _k12 = k_LY1 * k_LY2
    # zero branch written as _k12 * 0.0 so a NaN product propagates as in C
    pot_xfer12 = 1000 * jnp.where(_k12 > 0,
                                  jnp.sqrt(jnp.where(_k12 > 0, _k12, 1.0)),
                                  _k12 * 0.0) * (
        1e-9 * (prev[S.D_PSI_LY1] - prev[S.D_PSI_LY2])
        / (9.8 * 0.5 * (pars[P.LY1_z] + pars[P.LY2_z])) + 1)
    down12 = pot_xfer12 > 0
    SPACE_d = jnp.fmax(pars[P.LY2_z] * pars[P.LY2_por] * 1e3 - prev[S.H2O_LY2]
                       + (q_ly2 + fx[F.transp2]) * deltat, 0.0)
    H2O_d = jnp.fmax(prev[S.D_LF_LY1] * prev[S.H2O_LY1]
                     + (fx[F.infil] - q_ly1 - fx[F.evap] - fx[F.transp1]) * deltat,
                     0.0)
    MAX_d = prev[S.D_LF_LY1] * pot_xfer12 * DGCM_SEC_DAY * deltat
    ly1xly2_d = jnp.fmin(MAX_d, jnp.fmin(SPACE_d, H2O_d)) * one_over_deltat

    SPACE_u = jnp.fmax(pars[P.LY1_z] * pars[P.LY1_por] * 1e3 - prev[S.H2O_LY1]
                       - (fx[F.infil] - q_ly1 - fx[F.evap] - fx[F.transp1]) * deltat,
                       0.0)
    H2O_u = jnp.fmax(prev[S.D_LF_LY2] * prev[S.H2O_LY2]
                     - (q_ly2 + fx[F.transp2]) * deltat, 0.0)
    MAX_u = prev[S.D_LF_LY2] * pot_xfer12 * DGCM_SEC_DAY * deltat
    ly1xly2_u = -jnp.fmin(-MAX_u, jnp.fmin(SPACE_u, H2O_u)) * one_over_deltat

    fx[F.ly1xly2] = jnp.where(down12, ly1xly2_d, ly1xly2_u)
    TEMPxfer_1to2 = jnp.where(down12, prev[S.D_TEMP_LY1], prev[S.D_TEMP_LY2])

    # ---------------- LY2<->LY3 transfer (741-768)
    _k23 = k_LY2 * k_LY3
    pot_xfer23 = 1000 * jnp.where(_k23 > 0,
                                  jnp.sqrt(jnp.where(_k23 > 0, _k23, 1.0)),
                                  _k23 * 0.0) * (
        1e-9 * (prev[S.D_PSI_LY2] - prev[S.D_PSI_LY3])
        / (9.8 * 0.5 * (pars[P.LY2_z] + pars[P.LY3_z])) + 1)
    down23 = pot_xfer23 > 0
    SPACE_d3 = jnp.fmax(pars[P.LY3_z] * pars[P.LY3_por] * 1e3 - prev[S.H2O_LY3]
                        + q_ly3 * deltat, 0.0)
    H2O_d3 = jnp.fmax(prev[S.D_LF_LY2] * prev[S.H2O_LY2]
                      - (q_ly2 + fx[F.transp2]) * deltat, 0.0)
    MAX_d3 = prev[S.D_LF_LY2] * pot_xfer23 * DGCM_SEC_DAY * deltat
    ly2xly3_d = jnp.fmin(MAX_d3, jnp.fmin(SPACE_d3, H2O_d3)) * one_over_deltat

    SPACE_u3 = jnp.fmax(pars[P.LY2_z] * pars[P.LY2_por] * 1e3 - prev[S.H2O_LY2]
                        + (q_ly2 + fx[F.transp2]) * deltat, 0.0)
    H2O_u3 = jnp.fmax(prev[S.D_LF_LY3] * prev[S.H2O_LY3] - q_ly3 * deltat, 0.0)
    MAX_u3 = prev[S.D_LF_LY3] * pot_xfer23 * DGCM_SEC_DAY * deltat
    ly2xly3_u = -jnp.fmin(-MAX_u3, jnp.fmin(SPACE_u3, H2O_u3)) * one_over_deltat

    fx[F.ly2xly3] = jnp.where(down23, ly2xly3_d, ly2xly3_u)
    TEMPxfer_2to3 = jnp.where(down23, prev[S.D_TEMP_LY2], prev[S.D_TEMP_LY3])

    # ---------------- water pool updates + overflow (771-789)
    h2o_ly1 = prev[S.H2O_LY1] + (fx[F.infil] - fx[F.ly1xly2] - q_ly1
                                 - fx[F.evap] - fx[F.transp1]) * deltat
    h2o_ly2 = prev[S.H2O_LY2] + (fx[F.ly1xly2] - fx[F.ly2xly3] - q_ly2
                                 - fx[F.transp2]) * deltat
    h2o_ly3 = prev[S.H2O_LY3] + (fx[F.ly2xly3] - q_ly3) * deltat

    over1 = h2o_ly1 > LY1max
    q_ly1 = q_ly1 + jnp.where(over1, (h2o_ly1 - LY1max) * one_over_deltat, 0.0)
    h2o_ly1 = jnp.where(over1, LY1max, h2o_ly1)

    over2 = h2o_ly2 > LY2max
    # BUG_COMPAT: q_ly1_overflow — the C accumulates LY2 excess into q_ly1
    # (D1100.c:784); per-layer split wrong, ROFF total unaffected.
    q_ly1 = q_ly1 + jnp.where(over2, (h2o_ly2 - LY2max) * one_over_deltat, 0.0)
    h2o_ly2 = jnp.where(over2, LY2max, h2o_ly2)

    over3 = h2o_ly3 > LY3max
    q_ly3 = q_ly3 + jnp.where(over3, (h2o_ly3 - LY3max) * one_over_deltat, 0.0)
    h2o_ly3 = jnp.where(over3, LY3max, h2o_ly3)

    fx[F.q_ly1] = q_ly1   # post-overflow values, as read by the energy fluxes
    fx[F.q_ly2] = q_ly2
    fx[F.q_ly3] = q_ly3

    # ---------------- internal-energy fluxes (795-818)
    infiltemp = air_temp_k
    _iden = PREC - SNOWFALL + fx[F.melt]
    infiltemp = jnp.where(
        fx[F.melt] > 0,
        (infiltemp - DGCM_TK0C) * (PREC - SNOWFALL)
        / jnp.where(fx[F.melt] > 0, _iden, 1.0) + DGCM_TK0C,
        infiltemp)

    fx[F.infil_e] = fx[F.infil] * internal_energy_per_liquid_h2o_unit_mass(infiltemp)
    fx[F.evap_e] = fx[F.evap] * internal_energy_per_liquid_h2o_unit_mass(prev[S.D_TEMP_LY1])
    fx[F.transp1_e] = fx[F.transp1] * internal_energy_per_liquid_h2o_unit_mass(prev[S.D_TEMP_LY1])
    fx[F.transp2_e] = fx[F.transp2] * internal_energy_per_liquid_h2o_unit_mass(prev[S.D_TEMP_LY2])
    fx[F.ly1xly2_e] = fx[F.ly1xly2] * internal_energy_per_liquid_h2o_unit_mass(TEMPxfer_1to2)
    fx[F.ly2xly3_e] = fx[F.ly2xly3] * internal_energy_per_liquid_h2o_unit_mass(TEMPxfer_2to3)
    fx[F.q_ly1_e] = fx[F.q_ly1] * internal_energy_per_liquid_h2o_unit_mass(prev[S.D_TEMP_LY1])
    fx[F.q_ly2_e] = fx[F.q_ly2] * internal_energy_per_liquid_h2o_unit_mass(prev[S.D_TEMP_LY2])
    fx[F.q_ly3_e] = fx[F.q_ly3] * internal_energy_per_liquid_h2o_unit_mass(prev[S.D_TEMP_LY3])
    fx[F.ly1xly2_th_e] = 2 * pars[P.thermal_cond] \
        * (prev[S.D_TEMP_LY1] - prev[S.D_TEMP_LY2]) \
        / (pars[P.LY1_z] + pars[P.LY2_z]) * DGCM_SEC_DAY
    fx[F.ly2xly3_th_e] = 2 * pars[P.thermal_cond] \
        * (prev[S.D_TEMP_LY2] - prev[S.D_TEMP_LY3]) \
        / (pars[P.LY2_z] + pars[P.LY3_z]) * DGCM_SEC_DAY

    fx[F.geological] = jnp.asarray(PREDERIVED_GEO_FLUX, dtype=jnp.float64)
    e_ly1 = prev[S.E_LY1] + (fx[F.gh_in] + fx[F.infil_e] - fx[F.evap_e]
                             - fx[F.transp1_e] - fx[F.q_ly1_e]
                             - fx[F.ly1xly2_e] - fx[F.ly1xly2_th_e]) * deltat
    e_ly2 = prev[S.E_LY2] + (fx[F.ly1xly2_e] + fx[F.ly1xly2_th_e]
                             - fx[F.transp2_e] - fx[F.q_ly2_e]
                             - fx[F.ly2xly3_e] - fx[F.ly2xly3_th_e]) * deltat
    e_ly3 = prev[S.E_LY3] + (fx[F.ly2xly3_e] - fx[F.q_ly3_e]
                             + fx[F.ly2xly3_th_e] + fx[F.geological]) * deltat

    # ---------------- KNORR phenology (821-846)
    (lambda_next, knorr_T, laim, dlambdadt, f_T, f_d, lambda_tilde_max,
     lambda_W) = knorr_allocation(
        air_temp_k, deltat, 0.0, LAT, DOY, LAI, pars[P.lambda_max],
        pars[P.T_phi], pars[P.T_range], pars[P.plgr], pars[P.k_leaf],
        (prev[S.H2O_LY1] + h2o_ly1 + prev[S.H2O_LY2] + h2o_ly2) * 0.5,
        transp, pars[P.tau_W], pars[P.time_c], pars[P.time_r],
        prev[S.M_LAI_TEMP], prev[S.M_LAI_MAX])

    fx[F.target_LAI] = lambda_next
    fx[F.dlambda_dt] = dlambdadt * one_over_deltat
    fx[F.f_temp_thresh] = f_T
    fx[F.f_dayl_thresh] = f_d
    fx[F.lambda_tilde_max] = lambda_tilde_max
    fx[F.lambda_W] = lambda_W
    m_lai_max_new = laim
    m_lai_temp_new = knorr_T

    # ---------------- allocation + autotrophic respiration (850-905)
    ALLOC_FOL_POT = jnp.fmax(0.0, (fx[F.target_LAI] * pars[P.LCMA]
                                   - prev[S.C_fol]) * one_over_deltat)
    ALLOC_ROO_POT = jnp.fmax(0.0, (pars[P.phi_RL]
                                   * (fx[F.target_LAI] * pars[P.LCMA]))
                             * one_over_deltat)
    ALLOC_WOO_POT = jnp.fmax(0.0, (pars[P.phi_WL]
                                   * (fx[F.target_LAI] * pars[P.LCMA]))
                             * one_over_deltat)

    (F_LABPROD, _F_LABREL_ACTUAL, AUTO_RESP_MAINTENANCE, AUTO_RESP_GROWTH,
     ALLOC_FOL_ACTUAL, ALLOC_WOO_ACTUAL, ALLOC_ROO_ACTUAL, AUTO_RESP_TOTAL,
     _NPP, _CUE, NONLEAF_MORTALITY_FACTOR) = alloc_and_auto_resp_fluxes(
        deltat, air_temp_k, prev[S.C_woo], prev[S.C_roo], prev[S.C_lab],
        fx[F.gpp], liu_Rd, pars[P.rauto_mr_r], pars[P.rauto_mr_w],
        pars[P.rauto_gr], pars[P.rauto_mr_q10],
        ALLOC_FOL_POT, ALLOC_WOO_POT, ALLOC_ROO_POT)

    fx[F.nonleaf_mortality_factor] = NONLEAF_MORTALITY_FACTOR
    fx[F.resp_auto] = AUTO_RESP_TOTAL + liu_Rd
    fx[F.resp_auto_growth] = AUTO_RESP_GROWTH
    fx[F.resp_auto_maint] = AUTO_RESP_MAINTENANCE
    fx[F.resp_auto_maint_dark] = liu_Rd

    fx[F.ph_fol2lit] = jnp.where(fx[F.dlambda_dt] > 0, 0.0,
                                 -fx[F.dlambda_dt] * pars[P.LCMA])

    fx[F.lab_prod] = F_LABPROD
    fx[F.foliar_prod] = ALLOC_FOL_ACTUAL
    fx[F.root_prod] = ALLOC_ROO_ACTUAL
    fx[F.wood_prod] = ALLOC_WOO_ACTUAL

    # ---------------- heterotrophic respiration / methane (907-947)
    (aerobic_tr, anaerobic_tr, an_co2_ratio, an_ch4_ratio, _fT, _fV,
     _fW) = het_resp_rates_jcr(
        prev[S.D_TEMP_LY1], prev[S.D_SM_LY1], prev[S.D_LF_LY1],
        pars[P.S_fv], pars[P.thetas_opt], pars[P.fwc], pars[P.r_ch4],
        pars[P.Q10ch4], pars[P.Q10rhco2])

    fx[F.aetr] = aerobic_tr
    fx[F.antr] = anaerobic_tr
    fx[F.an_co2_c_ratio] = an_co2_ratio
    fx[F.an_ch4_c_ratio] = an_ch4_ratio

    ae_loss_cwd = prev[S.C_cwd] * aerobic_tr * pars[P.t_cwd]
    fx[F.ae_rh_cwd] = ae_loss_cwd * (1 - pars[P.tr_cwd2som])
    ae_loss_lit = prev[S.C_lit] * aerobic_tr * pars[P.t_lit]
    fx[F.ae_rh_lit] = ae_loss_lit * (1 - pars[P.tr_lit2som])
    fx[F.ae_rh_som] = prev[S.C_som] * aerobic_tr * pars[P.t_som]

    an_loss_cwd = prev[S.C_cwd] * anaerobic_tr * pars[P.t_cwd]
    fx[F.an_rh_cwd] = an_loss_cwd * (1 - pars[P.tr_cwd2som])
    an_loss_lit = prev[S.C_lit] * anaerobic_tr * pars[P.t_lit]
    fx[F.an_rh_lit] = an_loss_lit * (1 - pars[P.tr_lit2som])
    fx[F.an_rh_som] = prev[S.C_som] * anaerobic_tr * pars[P.t_som]
    fx[F.cwd2som] = (an_loss_cwd + ae_loss_cwd) * pars[P.tr_cwd2som]
    fx[F.lit2som] = (an_loss_lit + ae_loss_lit) * pars[P.tr_lit2som]
    fx[F.rh_co2] = (fx[F.an_rh_lit] + fx[F.an_rh_cwd] + fx[F.an_rh_som]) \
        * an_co2_ratio + (fx[F.ae_rh_lit] + fx[F.ae_rh_cwd] + fx[F.ae_rh_som])
    fx[F.rh_ch4] = (fx[F.an_rh_lit] + fx[F.an_rh_cwd] + fx[F.an_rh_som]) \
        * an_ch4_ratio

    # ---------------- carbon pool growth (951-962)
    c_lab = prev[S.C_lab] + (fx[F.gpp] - fx[F.Rd] - fx[F.resp_auto_maint]
                             - fx[F.foliar_prod] - fx[F.root_prod]
                             - fx[F.wood_prod] - fx[F.resp_auto_growth]) * deltat
    c_fol = prev[S.C_fol] + (fx[F.foliar_prod] - fx[F.ph_fol2lit]) * deltat
    c_roo = prev[S.C_roo] + fx[F.root_prod] * deltat
    c_woo = prev[S.C_woo] + fx[F.wood_prod] * deltat
    c_cwd = prev[S.C_cwd] - (fx[F.ae_rh_cwd] + fx[F.an_rh_cwd]
                             + fx[F.cwd2som]) * deltat
    c_lit = prev[S.C_lit] + (fx[F.ph_fol2lit] - fx[F.ae_rh_lit]
                             - fx[F.an_rh_lit] - fx[F.lit2som]) * deltat
    c_som = prev[S.C_som] + (fx[F.lit2som] - fx[F.ae_rh_som]
                             - fx[F.an_rh_som] + fx[F.cwd2som]) * deltat

    # ---------------- removals pass 1 of 4: disturbance (970-1001)
    TotalABGB = c_lab + c_fol + c_roo + c_woo
    DMF = DIST / TotalABGB
    CROPYIELD_factor = YIELD / TotalABGB

    fx[F.dist_lab] = c_lab * (2 * CROPYIELD_factor + DMF) * one_over_deltat
    fx[F.dist_fol] = c_fol * (2 * CROPYIELD_factor + DMF) * one_over_deltat
    fx[F.dist_roo] = c_roo * (2 * CROPYIELD_factor + DMF) * one_over_deltat
    fx[F.dist_woo] = c_woo * (2 * CROPYIELD_factor + DMF) * one_over_deltat

    fx[F.labyield2lit] = c_lab * CROPYIELD_factor * one_over_deltat
    fx[F.folyield2lit] = c_fol * CROPYIELD_factor * one_over_deltat
    fx[F.rooyield2lit] = c_roo * CROPYIELD_factor * one_over_deltat
    fx[F.wooyield2cwd] = c_woo * CROPYIELD_factor * one_over_deltat

    c_lab = c_lab - fx[F.dist_lab] * deltat
    c_fol = c_fol - fx[F.dist_fol] * deltat
    c_roo = c_roo - fx[F.dist_roo] * deltat
    c_woo = c_woo - fx[F.dist_woo] * deltat

    # ---------------- removals pass 2 of 4: fire combustion (1003-1024)
    CF_lab = pars[P.cf_ligneous]
    CF_fol = pars[P.cf_foliar]
    CF_roo = pars[P.cf_ligneous]
    CF_woo = pars[P.cf_ligneous]
    CF_cwd = pars[P.cf_ligneous]
    CF_lit = (pars[P.cf_foliar] + pars[P.cf_ligneous]) * 0.5
    CF_som = pars[P.cf_DOM]

    fx[F.f_lab] = c_lab * BURNED_AREA * CF_lab * one_over_deltat
    fx[F.f_fol] = c_fol * BURNED_AREA * CF_fol * one_over_deltat
    fx[F.f_roo] = c_roo * BURNED_AREA * CF_roo * one_over_deltat
    fx[F.f_woo] = c_woo * BURNED_AREA * CF_woo * one_over_deltat
    fx[F.f_cwd] = c_cwd * BURNED_AREA * CF_cwd * one_over_deltat
    fx[F.f_lit] = c_lit * BURNED_AREA * CF_lit * one_over_deltat
    fx[F.f_som] = c_som * BURNED_AREA * CF_som * one_over_deltat

    c_lab = c_lab - fx[F.f_lab] * deltat
    c_fol = c_fol - fx[F.f_fol] * deltat
    c_roo = c_roo - fx[F.f_roo] * deltat
    c_woo = c_woo - fx[F.f_woo] * deltat
    c_cwd = c_cwd - fx[F.f_cwd] * deltat
    c_lit = c_lit - fx[F.f_lit] * deltat
    c_som = c_som - fx[F.f_som] * deltat

    # ---------------- removals pass 3 of 4: aggregate mortality (1026-1049)
    AMF_C_lab = 1 - (1 - NONLEAF_MORTALITY_FACTOR) \
        * (1 - (BURNED_AREA * (1 - pars[P.resilience]))) * (1 - HMF)
    AMF_C_fol = 1 - (1 - LEAF_MORTALITY_FACTOR) \
        * (1 - (BURNED_AREA * (1 - pars[P.resilience]))) * (1 - HMF)
    AMF_C_roo = AMF_C_lab
    AMF_C_woo = AMF_C_lab

    fx[F.fx_lab2lit] = c_lab * AMF_C_lab * one_over_deltat
    fx[F.fx_fol2lit] = c_fol * AMF_C_fol * one_over_deltat
    fx[F.fx_roo2lit] = c_roo * AMF_C_roo * one_over_deltat
    fx[F.fx_woo2cwd] = c_woo * AMF_C_woo * one_over_deltat
    fx[F.fx_cwd2som] = c_cwd * BURNED_AREA * (1 - pars[P.resilience]) * one_over_deltat
    fx[F.fx_lit2som] = c_lit * BURNED_AREA * (1 - pars[P.resilience]) * one_over_deltat

    c_lab = c_lab - fx[F.fx_lab2lit] * deltat
    c_fol = c_fol - fx[F.fx_fol2lit] * deltat
    c_roo = c_roo - fx[F.fx_roo2lit] * deltat
    c_woo = c_woo - fx[F.fx_woo2cwd] * deltat

    # ---------------- removals pass 4 of 4: background mortality (1051-1069)
    fx[F.woo2cwd] = c_woo * pars[P.t_wood]
    fx[F.roo2lit] = c_roo * pars[P.t_root]
    fx[F.lab2lit] = c_lab * pars[P.t_lab]
    fx[F.fol2lit] = c_fol * pars[P.t_foliar]

    c_lab = c_lab - fx[F.lab2lit] * deltat
    c_fol = c_fol - fx[F.fol2lit] * deltat
    c_roo = c_roo - fx[F.roo2lit] * deltat
    c_woo = c_woo - fx[F.woo2cwd] * deltat

    # ---------------- dead-pool transfers part 2 of 2 (1071-1087)
    c_cwd = c_cwd + (fx[F.wooyield2cwd] + fx[F.woo2cwd] + fx[F.fx_woo2cwd]
                     - fx[F.fx_cwd2som]) * deltat
    c_lit = c_lit + (fx[F.labyield2lit] + fx[F.lab2lit] + fx[F.fx_lab2lit]
                     + fx[F.folyield2lit] + fx[F.fol2lit] + fx[F.fx_fol2lit]
                     + fx[F.rooyield2lit] + fx[F.roo2lit] + fx[F.fx_roo2lit]
                     - fx[F.fx_lit2som]) * deltat
    c_som = c_som + (fx[F.fx_cwd2som] + fx[F.fx_lit2som]) * deltat

    fx[F.f_total] = fx[F.f_lab] + fx[F.f_fol] + fx[F.f_roo] + fx[F.f_woo] \
        + fx[F.f_cwd] + fx[F.f_lit] + fx[F.f_som]
    fx[F.foliar_fire_frac] = BURNED_AREA * (CF_lab + (1 - CF_lab)
                                            * (1 - pars[P.resilience]))
    fx[F.lai_fire] = (prev[S.C_fol] / pars[P.LCMA]) * BURNED_AREA \
        * (CF_lab + (1 - CF_lab) * (1 - pars[P.resilience]))

    # ---------------- t+1 diagnostic states (1099-1134)
    d_lai = c_fol / pars[P.LCMA]
    d_scf = swe_new / (swe_new + pars[P.scf_scalar])

    T1, LF1 = soil_temp_and_liquid_frac(pars[P.LY1_vhc], pars[P.LY1_z],
                                        h2o_ly1, e_ly1)
    T2, LF2 = soil_temp_and_liquid_frac(pars[P.LY2_vhc], pars[P.LY2_z],
                                        h2o_ly2, e_ly2)
    T3, LF3 = soil_temp_and_liquid_frac(pars[P.LY3_vhc], pars[P.LY3_z],
                                        h2o_ly3, e_ly3)

    d_sm_ly1 = hydrofun_ewt2moi(h2o_ly1, pars[P.LY1_por], pars[P.LY1_z])
    d_sm_ly2 = hydrofun_ewt2moi(h2o_ly2, pars[P.LY2_por], pars[P.LY2_z])
    d_sm_ly3 = hydrofun_ewt2moi(h2o_ly3, pars[P.LY3_por], pars[P.LY3_z])
    d_psi_ly1 = _psi_clamped(d_sm_ly1, pars[P.retention])
    d_psi_ly2 = _psi_clamped(d_sm_ly2, pars[P.retention])
    d_psi_ly3 = _psi_clamped(d_sm_ly3, pars[P.retention])

    # ---------------- assemble outputs
    newp = [None] * NOPOOLS
    newp[S.C_lab], newp[S.C_fol], newp[S.C_roo], newp[S.C_woo] = c_lab, c_fol, c_roo, c_woo
    newp[S.C_cwd], newp[S.C_lit], newp[S.C_som] = c_cwd, c_lit, c_som
    newp[S.H2O_LY1], newp[S.H2O_LY2], newp[S.H2O_LY3] = h2o_ly1, h2o_ly2, h2o_ly3
    newp[S.H2O_SWE] = swe_new
    newp[S.E_LY1], newp[S.E_LY2], newp[S.E_LY3] = e_ly1, e_ly2, e_ly3
    newp[S.D_LAI], newp[S.D_SCF] = d_lai, d_scf
    newp[S.D_TEMP_LY1], newp[S.D_TEMP_LY2], newp[S.D_TEMP_LY3] = T1, T2, T3
    newp[S.D_LF_LY1], newp[S.D_LF_LY2], newp[S.D_LF_LY3] = LF1, LF2, LF3
    newp[S.D_SM_LY1], newp[S.D_SM_LY2], newp[S.D_SM_LY3] = d_sm_ly1, d_sm_ly2, d_sm_ly3
    newp[S.D_PSI_LY1], newp[S.D_PSI_LY2], newp[S.D_PSI_LY3] = d_psi_ly1, d_psi_ly2, d_psi_ly3
    newp[S.M_LAI_TEMP], newp[S.M_LAI_MAX] = m_lai_temp_new, m_lai_max_new
    assert all(x is not None for x in newp)
    new_pools = jnp.stack([jnp.asarray(x, dtype=jnp.float64) for x in newp])

    assert len(fx) == NOFLUXES, f"step wrote {len(fx)} of {NOFLUXES} fluxes"
    idx = jnp.array(sorted(fx), dtype=jnp.int32)
    vals = jnp.stack([jnp.asarray(fx[i], dtype=jnp.float64) for i in sorted(fx)])
    fluxes = jnp.zeros(NOFLUXES, dtype=jnp.float64).at[idx].set(vals)
    return new_pools, fluxes


def run_dalec_1100(pars, met: dict, LAT: float, deltat: float,
                   VegK: np.ndarray):
    """Full forward run. Returns (pools (T+1, 30), fluxes (T, 100)).

    Reproduces the C isfinite break (D1100.c:1137-1141) under the
    per-sample-zeroed-buffer canonical form: the breaking step's non-finite
    values ARE recorded; every later step is exactly zero.
    """
    pars = jnp.asarray(pars, dtype=jnp.float64)
    T = len(met["SSRD"])
    xs = jnp.stack([jnp.asarray(met[k], dtype=jnp.float64)
                    for k in MET_COLUMNS], axis=1)
    xs = jnp.concatenate([xs, jnp.asarray(VegK, dtype=jnp.float64)[:, None]],
                         axis=1)
    p0 = _initial_pools(pars)

    def scan_step(carry, x):
        prev, alive = carry
        met_row, vegk_n = x[:13], x[13]
        new_pools, fluxes = _step(pars, deltat, LAT, prev, met_row, vegk_n)
        out_pools = jnp.where(alive, new_pools, jnp.zeros(NOPOOLS))
        out_fluxes = jnp.where(alive, fluxes, jnp.zeros(NOFLUXES))
        new_alive = jnp.logical_and(
            alive, jnp.isfinite(new_pools[:N_PROGNOSTIC]).all())
        # carry the emitted (possibly zeroed) pools: after a break the C
        # leaves the remaining steps at calloc-zero and never reads them
        return (out_pools, new_alive), (out_pools, out_fluxes)

    (_, _), (pools_tail, fluxes) = lax.scan(
        scan_step, (p0, jnp.array(True)), xs)
    pools = jnp.concatenate([p0[None, :], pools_tail], axis=0)
    return pools, fluxes
