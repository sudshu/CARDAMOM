"""LIU_AN_ET_REFACTOR.c — Farquhar/Medlyn photosynthesis + PM transpiration.

Order-critical detail preserved from C: SRAD is REASSIGNED at line 221
(albedo-scaled) before the `beta_factor > 0 && SRAD > 0` branch at line 225,
so the branch tests the scaled value. gs = 0 (An <= 0) drives the C through
1/gs = inf and transp = petVnum/inf = 0; the where-pair reproduces the same
selected values.
"""
import jax.numpy as jnp

from ..constants import DGCM_TK0C
from .ad_guards import inv_exp_clamped


def liu_an_et(SRAD, VPD, TEMP, vcmax25, co2, beta_factor, g1, LAI, ga, VegK,
              Tupp, Tdown, C3_frac, clumping, leaf_refl_par, leaf_refl_nir,
              maxPevap, precip, q10canopy, q10canopyRd, canopyRdsf, NSC,
              deltat):
    Ephoton = 2.0e-25 / 500.0e-9
    NA = 6.02e23
    lambda0 = 2.26e6
    gammaV = 100 * 1005 / (lambda0 * 0.622)

    PAR = SRAD / (2 * Ephoton * NA) * 1e6
    canopy_scale = (1. - jnp.exp(-VegK * LAI * clumping)) / VegK
    PAR = PAR * ((1. - leaf_refl_par) * (1. - jnp.exp(-VegK * LAI * clumping)))

    T_C = TEMP - DGCM_TK0C

    Kc = 300. * jnp.exp(0.074 * (T_C - 25.))
    Ko = 300. * jnp.exp(0.015 * (T_C - 25.))
    cp = 36.9 + 1.18 * (T_C - 25.) + 0.036 * jnp.power(T_C - 25., 2.)

    q_10 = q10canopy
    fT = jnp.power(q10canopyRd, (T_C - 25.) / 10)

    Vcmax = (vcmax25 * jnp.power(q_10, 0.1 * (T_C - 25.))
             / ((1 + jnp.exp(0.3 * (T_C - (Tupp - DGCM_TK0C))))
                * (1 + jnp.exp(0.3 * ((Tdown - DGCM_TK0C) - T_C)))))
    Jmax = Vcmax * jnp.exp(1.)
    J = (0.3 * PAR + Jmax
         - jnp.sqrt(jnp.power(0.3 * PAR + Jmax, 2)
                    - 4. * 0.9 * 0.3 * PAR * Jmax)) / 2. / 0.9

    medlyn_term = 1. + g1 / jnp.sqrt(VPD)
    ci = co2 * (1. - 1. / medlyn_term)
    ci = jnp.where(ci < cp, cp, ci)          # CLM 4.5 clamp, C line 147

    a1 = Vcmax * (ci - cp) / (ci + Kc * (1. + 209. / Ko))
    a2 = J * (ci - cp) / (4. * (ci + 2. * cp))
    Ag_C3 = jnp.fmin(a1 * beta_factor, a2)
    Rd_C3 = canopyRdsf * vcmax25 * fT

    a1 = Vcmax
    a2 = J
    Ag_C4 = jnp.fmin(a1 * beta_factor, a2)
    Rd_C4 = canopyRdsf * vcmax25 * fT

    Ag = C3_frac * Ag_C3 + (1. - C3_frac) * Ag_C4
    Rd = C3_frac * Rd_C3 + (1. - C3_frac) * Rd_C4

    Rd_daily_potential = Rd * canopy_scale * (12.e-6) * (24. * 60. * 60.)
    # GRADIENT HARDENING (value-identical, bit-exact): exp(x) overflows to
    # inf exactly for x > log(DBL_MAX) = 709.782712893384, where the C's
    # 1/exp(x) becomes +0.0. Below the cutoff the operand passes through
    # fmin unchanged (identical bits); above it we select the same +0.0 —
    # but exp stays finite, so no inf/inf NaN reaches the backward pass.
    LOG_DBL_MAX = 709.782712893384
    x_lmf = NSC / (jnp.where(Rd_daily_potential == 0, 1.0,
                             Rd_daily_potential) * deltat)
    # SECOND-ORDER LEAK GUARD: inv_exp_clamped is bit-identical to
    # 1/exp(min(x, LOG_DBL_MAX)); its JVP -w*dx keeps forward tangents
    # finite for x just below the cutoff (exp(x)*dx would overflow and
    # give inf/inf = NaN in jvp/HVP; see ad_guards.py).
    LEAF_MORTALITY_FACTOR = jnp.where(
        Rd_daily_potential == 0,
        0.0,
        jnp.where(x_lmf > LOG_DBL_MAX, 0.0, inv_exp_clamped(x_lmf)))
    OUT_Rd = Rd_daily_potential * (1 - LEAF_MORTALITY_FACTOR)
    Rd = Rd * (1 - LEAF_MORTALITY_FACTOR)

    An = Ag - Rd

    OUT_Ag = Ag * canopy_scale * (12.e-6) * (24. * 60. * 60.)
    OUT_An = An * canopy_scale * (12.e-6) * (24. * 60. * 60.)

    # ------------- transpiration / evaporation -------------
    VPD_kPa = VPD
    sV = 0.04145 * jnp.exp(0.06088 * T_C)

    SRADg = (1. - 0.5 * (leaf_refl_par + leaf_refl_nir)) * SRAD \
        * jnp.exp(-VegK * LAI * clumping)
    SRAD = (1. - 0.5 * (leaf_refl_par + leaf_refl_nir)) * SRAD  # C line 221

    petVnum = (sV * (SRAD - SRADg)
               + 1.225 * 1005 * VPD_kPa * ga) / lambda0 * 60 * 60
    petVnumB = 1.26 * (sV * SRADg) / (sV + gammaV) / lambda0 * 60 * 60

    gs = jnp.fmax(0.0, 1.6 * An / (co2 - ci) * LAI * 0.02405)
    # GRADIENT HARDENING: gs == 0 drives the C through 1/gs = inf and
    # transp = petVnum/inf = ±0. We divide by a safe gs and select 0.0 for
    # that case (|Δ| = 0; only the sign of zero can differ from the C),
    # keeping the backward pass NaN-free.
    gs_pos = gs > 0
    transp_active = petVnum / (sV + gammaV * (ga * (1 / ga
                                                    + 1 / jnp.where(gs_pos, gs, 1.0))))
    transp = jnp.where(jnp.logical_and(beta_factor > 0, SRAD > 0),
                       jnp.where(gs_pos, transp_active, 0.0), 0.0)
    OUT_transp = transp * 24

    evap_scale_factpr = jnp.fmin(precip / maxPevap, 1.)
    evap = petVnumB * evap_scale_factpr
    OUT_evap = evap * 24

    return OUT_An, OUT_Ag, OUT_Rd, OUT_transp, OUT_evap, LEAF_MORTALITY_FACTOR
