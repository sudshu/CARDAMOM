"""Observation operators + 31-term likelihood + DALEC_MLF2 composition.

Transcribed from DALEC_OBSERVATION_OPERATORS.c, DALEC_ALL_LIKELIHOOD.c and
CARDAMOM_LIKELIHOOD_FUNCTION.c. All observation CONFIG is static python
(from data_prep.CbfData); only model pools/fluxes/pars are traced.

Reproduced C quirks (BUG_COMPAT):
- OBSOPE.rhch4_rhco2_flux = F.rh_ch4 / F.rh_co2 — C INTEGER DIVISION of two
  flux indices, then used as a PARS index (obsope_int_div_index).
- OBSOPE.C3frac_PARAM is never assigned (commented out) — static struct
  zero-init makes it 0, so PEQ_C3frac reads pars[0] (obsope_c3frac_unset).
- SINGLE_OBS validobs garbage (obs_minmax_uninit) cannot affect any ML term:
  the likelihood gates on value != -9999 only, which we replicate.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from ..indices import F, NOEDCS, P, S
from .data_prep import DEFAULT, CbfData, SingleObs, TimeseriesObs

NOLIKELIHOODS = 31

# LIKELIHOOD_INDICES (DALEC_ALL_LIKELIHOOD.c:21-57)
LI = {name: i for i, name in enumerate((
    "ABGB", "CH4", "DOM", "ET", "LE", "H", "EWT", "GPP", "SIF", "LAI",
    "NBE", "ROFF", "SCF", "FIR", "SWE",
    "Mean_ABGB", "Mean_FIR", "Mean_GPP", "Mean_LAI",
    "PEQ_Cefficiency", "PEQ_CUE", "PEQ_NBEmrg", "PEQ_iniSnow", "PEQ_iniSOM",
    "PEQ_C3frac", "PEQ_Vcmax25", "PEQ_LCMA", "PEQ_clumping",
    "PEQ_r_ch4", "PEQ_S_fv", "PEQ_rhch4_rhco2"))}


def _mid(pools, pool_idx):
    """(POOLS[n] + POOLS[n+1]) * 0.5 series for one pool."""
    return (pools[:-1, pool_idx] + pools[1:, pool_idx]) * 0.5


def timeseries_likelihood(obs: TimeseriesObs, mod_full) -> jnp.ndarray:
    """CARDAMOM_TIMESERIES_OBS_LIKELIHOOD. mod_full is the length-T model
    series; gathering, filtering and unc handling follow the C exactly.
    Static config -> python branches; traced values -> jnp."""
    if obs.valid_obs_length == 0:
        return jnp.asarray(0.0)

    vi = obs.valid_idx
    N = len(vi)
    mod = mod_full[vi]
    obsv = jnp.asarray(obs.values[vi])
    unc = jnp.asarray(obs.unc[vi])

    smean, smonth = obs.single_mean_unc, obs.single_monthly_unc
    sann, sdec, strd = obs.single_annual_unc, obs.single_decadal_unc, obs.trend_unc

    if obs.opt_normalization > 0:
        mean_mod = jnp.sum(mod) / N
        mean_obs = jnp.sum(obsv) / N
        if obs.opt_normalization == 1:
            mod, obsv = mod - mean_mod, obsv - mean_obs
        else:
            mod, obsv = mod / mean_mod, obsv / mean_obs

    if not np.isinf(obs.min_threshold):
        # C max macro (a<b ? b : a): NaN-propagating like jnp.maximum
        mod = jnp.maximum(mod, obs.min_threshold)
        obsv = jnp.maximum(obsv, obs.min_threshold)

    if obs.opt_unc_type == 1:
        mod, obsv, unc = jnp.log(mod), jnp.log(obsv), jnp.log(unc)
        smean, smonth = np.log(smean), np.log(smonth)
        sann, sdec = np.log(sann), np.log(sdec)

    f = obs.opt_filter
    tot = jnp.asarray(0.0)
    if f == 0:
        tot = jnp.sum(jnp.power((mod - obsv) / unc, 2))
    elif f == 1:
        tot = jnp.power((jnp.sum(mod) / N - jnp.sum(obsv) / N) / smean, 2)
    elif f in (2, 3):
        ny = N // 12
        m2 = mod[:ny * 12].reshape(ny, 12)
        o2 = obsv[:ny * 12].reshape(ny, 12)
        mam = jnp.sum(m2, axis=1) / 12
        oam = jnp.sum(o2, axis=1) / 12
        if f == 2:
            tot = tot + jnp.sum(jnp.power(
                (m2 - o2 - mam[:, None] + oam[:, None]) / smonth, 2))
        tot = tot + jnp.sum(jnp.power((oam - mam) / sann, 2))
    elif f in (4, 5):
        icm = np.asarray(vi) % 12
        counts = np.bincount(icm, minlength=12)
        modcm = jnp.zeros(12).at[icm].add(mod)
        obscm = jnp.zeros(12).at[icm].add(obsv)
        nz = counts > 0
        modcm = jnp.where(nz, modcm / np.maximum(counts, 1), modcm)
        obscm = jnp.where(nz, obscm / np.maximum(counts, 1), obscm)
        tot = tot + jnp.sum(jnp.where(
            nz, jnp.power((modcm - obscm) / smonth, 2), 0.0))
        if f == 4:
            multi = counts[icm] > 1
            tot = tot + jnp.sum(jnp.where(
                multi,
                jnp.power((mod - modcm[icm] - obsv + obscm[icm]) / sann, 2),
                0.0))
    elif f == 6:
        ny = N // 12
        for m in range(1, ny - 1):
            sl = slice((m - 1) * 12, (m - 1) * 12 + 36)
            m3 = jnp.sum(mod[sl]) / 36
            o3 = jnp.sum(obsv[sl]) / 36
            tot = tot + jnp.power((o3 - m3) / sann, 2)
    elif f == 7:
        tmm = jnp.sum(mod) / N
        tmo = jnp.sum(obsv) / N
        tot = tot + jnp.power((tmo - tmm) / smean, 2)
        m1, o1 = jnp.sum(mod[:120]) / 120, jnp.sum(obsv[:120]) / 120
        m2_, o2_ = jnp.sum(mod[120:]) / (N - 120), jnp.sum(obsv[120:]) / (N - 120)
        tot = tot + jnp.power(((o1 - tmo) - (m1 - tmm)) / sdec, 2)
        tot = tot + jnp.power(((o2_ - tmo) - (m2_ - tmm)) / sdec, 2)
        for y in range(10):
            sl = slice(y * 12, y * 12 + 12)
            ma, oa = jnp.sum(mod[sl]) / 12, jnp.sum(obsv[sl]) / 12
            tot = tot + jnp.power(((oa - o1) - (ma - m1)) / sann, 2)
        for y in range(10, N // 12):
            sl = slice(y * 12, y * 12 + 12)
            ma, oa = jnp.sum(mod[sl]) / 12, jnp.sum(obsv[sl]) / 12
            tot = tot + jnp.power(((oa - o2_) - (ma - m2_)) / sann, 2)
    elif f == 8:
        mm, mo = jnp.sum(mod) / N, jnp.sum(obsv) / N
        tot = tot + jnp.power((mm - mo) / smean, 2)
        ny = N // 12
        m2 = mod[:ny * 12].reshape(ny, 12)
        o2 = obsv[:ny * 12].reshape(ny, 12)
        mam = jnp.sum(m2, axis=1) / 12
        oam = jnp.sum(o2, axis=1) / 12
        tot = tot + jnp.sum(jnp.power(((oam - mo) - (mam - mm)) / sann, 2))
        tot = tot + jnp.sum(jnp.power(
            ((m2 - mam[:, None]) - (o2 - oam[:, None])) / smonth, 2))
    elif f == 9:
        tot = jnp.sum(jnp.power((mod - obsv) / unc, 2))
        if (N // 12) % 2 == 0:
            nmonths, offset = N // 2, 0
        else:
            nmonths, offset = N // 2 - 6, 12
        m1 = jnp.sum(mod[:nmonths]) / nmonths
        o1 = jnp.sum(obsv[:nmonths]) / nmonths
        m2_ = jnp.sum(mod[nmonths + offset:]) / nmonths
        o2_ = jnp.sum(obsv[nmonths + offset:]) / nmonths
        tot = tot + jnp.power((m1 - m2_ - o1 + o2_) / strd, 2)

    return -0.5 * tot


def single_likelihood(sobs: SingleObs, mod) -> jnp.ndarray:
    if not sobs.active:
        return jnp.asarray(0.0)
    obs = sobs.value
    unc = sobs.unc
    if not np.isinf(sobs.min_threshold):
        mod = jnp.maximum(mod, sobs.min_threshold)
        obs = max(obs, sobs.min_threshold)
    if sobs.opt_unc_type == 1:
        mod, obs, unc = jnp.log(mod), np.log(obs), np.log(unc)
    return -0.5 * jnp.power((mod - obs) / unc, 2)


def likelihood(cbf: CbfData, pars, pools, fluxes) -> tuple:
    """DALEC_ALL_LIKELIHOOD's LIKELIHOOD(): (ML (31,), P). All SUPPORT_*
    flags are true for 1100 except CUE/Cefficiency (MODCONFIG:1447-1483)."""
    T = cbf.n_timesteps
    ts, single = cbf.ts, cbf.single
    ml = {}

    # ---- timeseries operators + terms (only streams the C computes)
    if ts["ABGB"].valid_obs_length > 0 or single["Mean_ABGB"].value != DEFAULT:
        m_abgb = (_mid(pools, S.C_lab) + _mid(pools, S.C_fol)
                  + _mid(pools, S.C_roo) + _mid(pools, S.C_woo))
    else:
        m_abgb = jnp.zeros(T)
    ml[LI["ABGB"]] = timeseries_likelihood(ts["ABGB"], m_abgb)
    ml[LI["CH4"]] = timeseries_likelihood(ts["CH4"], fluxes[:, F.rh_ch4])
    # DOM operator exists but DOM obs absent in practice; series only if valid
    if ts["DOM"].valid_obs_length > 0:
        m_dom = (_mid(pools, S.C_cwd) + _mid(pools, S.C_lit)
                 + _mid(pools, S.C_som))
        ml[LI["DOM"]] = timeseries_likelihood(ts["DOM"], m_dom)
    else:
        ml[LI["DOM"]] = jnp.asarray(0.0)
    ml[LI["ET"]] = timeseries_likelihood(ts["ET"], fluxes[:, F.ets])
    ml[LI["LE"]] = timeseries_likelihood(ts["LE"], fluxes[:, F.latent_heat])
    ml[LI["H"]] = timeseries_likelihood(ts["H"], fluxes[:, F.sensible_heat])
    if ts["EWT"].valid_obs_length > 0:
        m_ewt = (_mid(pools, S.H2O_LY1) + _mid(pools, S.H2O_LY2)
                 + _mid(pools, S.H2O_LY3) + _mid(pools, S.H2O_SWE))
        ml[LI["EWT"]] = timeseries_likelihood(ts["EWT"], m_ewt)
    else:
        ml[LI["EWT"]] = jnp.asarray(0.0)
    ml[LI["GPP"]] = timeseries_likelihood(ts["GPP"], fluxes[:, F.gpp])
    ml[LI["SIF"]] = timeseries_likelihood(ts["SIF"], fluxes[:, F.gpp])
    ml[LI["LAI"]] = timeseries_likelihood(ts["LAI"], _mid(pools, S.D_LAI))
    m_nbe = (-fluxes[:, F.gpp] + fluxes[:, F.resp_auto]
             + fluxes[:, F.rh_co2] + fluxes[:, F.f_total])
    ml[LI["NBE"]] = timeseries_likelihood(ts["NBE"], m_nbe)
    m_roff = (fluxes[:, F.q_ly1] + fluxes[:, F.q_ly2] + fluxes[:, F.q_ly3]
              + fluxes[:, F.q_surf])
    ml[LI["ROFF"]] = timeseries_likelihood(ts["ROFF"], m_roff)
    ml[LI["SCF"]] = timeseries_likelihood(ts["SCF"], _mid(pools, S.D_SCF))
    ml[LI["FIR"]] = timeseries_likelihood(ts["FIR"], fluxes[:, F.f_total])
    ml[LI["SWE"]] = timeseries_likelihood(ts["SWE"], _mid(pools, S.H2O_SWE))

    # ---- single-obs terms
    ml[LI["Mean_ABGB"]] = single_likelihood(single["Mean_ABGB"],
                                            jnp.sum(m_abgb) / T)
    ml[LI["Mean_FIR"]] = single_likelihood(single["Mean_FIR"],
                                           jnp.sum(fluxes[:, F.f_total]) / T)
    ml[LI["Mean_GPP"]] = single_likelihood(single["Mean_GPP"],
                                           jnp.sum(fluxes[:, F.gpp]) / T)
    ml[LI["Mean_LAI"]] = single_likelihood(single["Mean_LAI"],
                                           jnp.sum(_mid(pools, S.D_LAI)) / T)

    # PEQ terms (SUPPORT flags per MODCONFIG; CUE comes via CUEmrg)
    mgpp = jnp.sum(fluxes[:, F.gpp]) / T
    mrauto = jnp.sum(fluxes[:, F.resp_auto]) / T
    ml[LI["PEQ_CUE"]] = single_likelihood(single["PEQ_CUE"], 1 - mrauto / mgpp)
    mrhet = jnp.sum(fluxes[:, F.rh_co2]) / T
    mfire = jnp.sum(fluxes[:, F.f_total]) / T
    ml[LI["PEQ_NBEmrg"]] = single_likelihood(single["PEQ_NBEmrg"],
                                             mgpp / (mrauto + mrhet + mfire))
    ml[LI["PEQ_Cefficiency"]] = jnp.asarray(0.0)   # SUPPORT flag false
    ml[LI["PEQ_iniSnow"]] = single_likelihood(single["PEQ_iniSnow"],
                                              pars[P.i_SWE])
    ml[LI["PEQ_iniSOM"]] = single_likelihood(single["PEQ_iniSOM"],
                                             pars[P.i_som])
    # BUG_COMPAT obsope_c3frac_unset: C3frac_PARAM never set -> pars[0]
    ml[LI["PEQ_C3frac"]] = single_likelihood(single["PEQ_C3frac"], pars[0])
    ml[LI["PEQ_Vcmax25"]] = single_likelihood(single["PEQ_Vcmax25"],
                                              pars[P.Vcmax25])
    ml[LI["PEQ_LCMA"]] = single_likelihood(single["PEQ_LCMA"], pars[P.LCMA])
    ml[LI["PEQ_clumping"]] = single_likelihood(single["PEQ_clumping"],
                                               pars[P.clumping])
    ml[LI["PEQ_r_ch4"]] = single_likelihood(single["PEQ_r_ch4"],
                                            pars[P.r_ch4])
    ml[LI["PEQ_S_fv"]] = single_likelihood(single["PEQ_S_fv"], pars[P.S_fv])
    # BUG_COMPAT obsope_int_div_index: index = F.rh_ch4 // F.rh_co2 (C int
    # division of flux indices), then 1 - pars[that index]
    ml[LI["PEQ_rhch4_rhco2"]] = single_likelihood(
        single["PEQ_rhch4_rhco2"], 1 - pars[F.rh_ch4 // F.rh_co2])

    ML = jnp.stack([ml[i] for i in range(NOLIKELIHOODS)])
    Psum = jnp.sum(ML)
    Psum = jnp.where(jnp.isnan(Psum), -jnp.inf, Psum)  # isnan(P) -> log(0)
    return ML, Psum


def mlf2(cbf: CbfData, edc_cfg: dict, pars, pools, fluxes):
    """DALEC_MLF2 composition under jit (model already run by caller).

    Returns (M_EDCs (15,), M_LIKELIHOODS (31,), P) with the C's gate
    arithmetic: prerun EDCs -> [model] -> postrun EDCs (if P>-inf) ->
    likelihood (if P>-inf); gated stages leave zeroed records.
    """
    from .. import edcs as edcs_mod

    vals = edcs_mod.edc_values(pars, pools, fluxes, edc_cfg)
    rec_pre, p_pre = edcs_mod.record_phase(vals, True, jnp.asarray(0.0))
    P = p_pre if cbf.EDC == 1 else jnp.asarray(0.0)

    proceed = P > -jnp.inf
    rec_post, p_post = edcs_mod.record_phase(vals, False, jnp.asarray(0.0))
    if cbf.EDC == 1:
        rec = rec_pre + jnp.where(proceed, rec_post, jnp.zeros(NOEDCS))
        P = jnp.where(proceed, P + p_post, P)
    else:
        rec = rec_pre

    ML, Plik = likelihood(cbf, pars, pools, fluxes)
    proceed2 = P > -jnp.inf
    ML = jnp.where(proceed2, ML, jnp.zeros(NOLIKELIHOODS))
    P = jnp.where(proceed2, P + Plik, P)
    return rec, ML, P
