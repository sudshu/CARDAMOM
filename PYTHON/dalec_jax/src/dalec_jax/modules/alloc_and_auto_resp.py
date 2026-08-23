"""ALLOC_AND_AUTO_RESP_FLUXES.c — NSC-limited maintenance respiration,
exponential growth-factor mobilization, allocation, growth respiration.

Two C zero-guards become where-pairs whose discarded branches are IEEE
value-safe (x/0 and exp overflow are dropped by where): the
POTENTIAL_AUTO_RESP_MAINTENANCE==0 branch and the F_LABREL_DEMAND!=0 branch.
CUE = NPP/GPP is UNGUARDED in C (GPP=0 → ±inf/NaN) and stays unguarded here.
"""
import jax.numpy as jnp

from ..constants import DGCM_TK0C


def alloc_and_auto_resp_fluxes(deltat, TEMP, C_LIVE_W, C_LIVE_R, NSC, GPP, Rd,
                               mr_r, mr_w, gr, Q10mr,
                               ALLOC_FOL_POT, ALLOC_WOO_POT, ALLOC_ROO_POT):
    fT = jnp.power(Q10mr, (TEMP - (25 + DGCM_TK0C)) / 10)
    POTENTIAL_AUTO_RESP_MAINTENANCE = mr_w * fT * C_LIVE_W + mr_r * fT * C_LIVE_R

    F_LABPROD = GPP - Rd
    NSC_PLUS_GPP_RATE = NSC / deltat + (GPP - Rd)

    # GRADIENT HARDENING (value-identical, bit-exact): safe division operand
    # plus the exact exp-overflow cutoff (see liu_an_et.py) — the C's
    # 1/exp(x) is +0.0 above log(DBL_MAX) and unchanged bits below it.
    LOG_DBL_MAX = 709.782712893384
    x_nmf = NSC_PLUS_GPP_RATE / jnp.where(
        POTENTIAL_AUTO_RESP_MAINTENANCE == 0, 1.0,
        POTENTIAL_AUTO_RESP_MAINTENANCE)
    NONLEAF_MORTALITY_FACTOR = jnp.where(
        POTENTIAL_AUTO_RESP_MAINTENANCE == 0,
        0.0,
        jnp.where(x_nmf > LOG_DBL_MAX, 0.0,
                  1 / jnp.exp(jnp.minimum(x_nmf, LOG_DBL_MAX))))

    AUTO_RESP_MAINTENANCE = (POTENTIAL_AUTO_RESP_MAINTENANCE
                             * (1 - NONLEAF_MORTALITY_FACTOR))

    LEFTOVER_NSC_RATE = NSC_PLUS_GPP_RATE - AUTO_RESP_MAINTENANCE

    F_LABREL_SUPPLY = jnp.fmax(0.0, gr * LEFTOVER_NSC_RATE)
    TOTAL_GROWTH_POT = ALLOC_FOL_POT + ALLOC_WOO_POT + ALLOC_ROO_POT
    F_LABREL_DEMAND = jnp.fmax(0.0, TOTAL_GROWTH_POT)

    x_gf = F_LABREL_SUPPLY / jnp.where(F_LABREL_DEMAND != 0,
                                       F_LABREL_DEMAND, 1.0)
    GF = jnp.where(F_LABREL_DEMAND != 0,
                   jnp.where(x_gf > LOG_DBL_MAX, 0.0,
                             1 / jnp.exp(jnp.minimum(x_gf, LOG_DBL_MAX))),
                   0.0)

    F_LABREL_ACTUAL = F_LABREL_DEMAND * (1 - GF)

    ALLOC_FOL_ACTUAL = ALLOC_FOL_POT * (1 - GF)
    ALLOC_WOO_ACTUAL = ALLOC_WOO_POT * (1 - GF)
    ALLOC_ROO_ACTUAL = ALLOC_ROO_POT * (1 - GF)

    TOTAL_GROWTH_ACTUAL = ALLOC_FOL_ACTUAL + ALLOC_WOO_ACTUAL + ALLOC_ROO_ACTUAL
    AUTO_RESP_GROWTH = (1 - gr) / gr * TOTAL_GROWTH_ACTUAL

    AUTO_RESP_TOTAL = AUTO_RESP_MAINTENANCE + AUTO_RESP_GROWTH
    NPP = GPP - AUTO_RESP_TOTAL
    CUE = NPP / GPP

    return (F_LABPROD, F_LABREL_ACTUAL, AUTO_RESP_MAINTENANCE,
            AUTO_RESP_GROWTH, ALLOC_FOL_ACTUAL, ALLOC_WOO_ACTUAL,
            ALLOC_ROO_ACTUAL, AUTO_RESP_TOTAL, NPP, CUE,
            NONLEAF_MORTALITY_FACTOR)
