"""HET_RESP_RATES_JCR.c — joint CO2/CH4 heterotrophic respiration scalars.

The SM=1 corner is deliberate IEEE flow-through: fV = fmax(0, 1-pow(1,S_fv))
= 0, theta_ae = (0)/0 + 1 = NaN, and both fW branch conditions evaluate False
on NaN, selecting fW = 0 — identical to the C's if/else-if/else. No guard is
added because a guard could not change the selected value but WOULD change
the discarded branch; gradients get hardened in P6 without touching values.
"""
import jax.numpy as jnp


def het_resp_rates_jcr(TEMP, SM, LF, S_FV, SM_OPT, FWC, R_CH4, Q10CH4, Q10CO2):
    reftemp = 298.15
    thetas = SM

    fT = jnp.power(Q10CO2, (TEMP - reftemp) / 10) * LF
    fV = jnp.fmax(0.0, (1 - jnp.power(thetas, S_FV)))

    # GRADIENT HARDENING (value-identical by construction): the C computes
    # theta_ae = (thetas-1)/fV + 1, which is NaN/inf when fV == 0; both fW
    # branch conditions then evaluate False and fW = 0. We divide by a safe
    # operand (exactly fV whenever fV > 0) and fold "fV > 0" into the branch
    # conditions, so every selected VALUE is bit-identical to the C while no
    # NaN can enter the backward pass.
    fV_pos = fV > 0
    theta_ae = ((thetas - 1) / jnp.where(fV_pos, fV, 1.0) + 1)
    fW1 = 1 / SM_OPT * theta_ae
    fW2 = ((1 - FWC) / (SM_OPT - 1) * theta_ae + (FWC - (1 - FWC) / (SM_OPT - 1)))
    c1 = fV_pos & (theta_ae >= 0) & (theta_ae < SM_OPT)
    c2 = fV_pos & (theta_ae >= SM_OPT) & (theta_ae <= 1)
    fW = jnp.where(c1, fW1, jnp.where(c2, fW2, 0.0))

    fT_ch4 = jnp.power(Q10CH4, (TEMP - reftemp) / 10) * LF
    fCH4 = jnp.fmin(R_CH4 * fT_ch4, 1.0)

    aerobic_tr = fW * fT * fV
    anaerobic_tr = FWC * fT * (1 - fV)
    anaerobic_co2_c_ratio = 1 - fCH4
    anaerobic_ch4_c_ratio = fCH4
    return (aerobic_tr, anaerobic_tr, anaerobic_co2_c_ratio,
            anaerobic_ch4_c_ratio, fT, fV, fW)
