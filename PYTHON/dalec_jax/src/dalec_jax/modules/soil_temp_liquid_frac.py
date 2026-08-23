"""SOIL_TEMP_AND_LIQUID_FRAC.c — internal energy → (temperature, liquid frac).

Three-branch select on internal energy vs the frozen (UI3) and fully-liquid
(UL3) thresholds; branch conditions are STRICT inequalities exactly as in C
(lines 21/23), so the plateau branch takes both boundary values. The
unfrozen/frozen expressions are computed for all inputs (both-branch
evaluation) — division by UI3 or soil_water can produce inf/NaN in the
not-selected branch, which jnp.where discards, matching C values exactly.
"""
import jax.numpy as jnp

from ..constants import (DGCM_LATENT_HEAT_FUSION_3, DGCM_SPECIFIC_HEAT_ICE,
                         DGCM_SPECIFIC_HEAT_WATER, DGCM_T3,
                         DGCM_T_LIQUID_H2O_ZERO_ENERGY, DGCM_TK0C)


def soil_temp_and_liquid_frac(dry_soil_vol_heat_capacity, depth, soil_water,
                              internal_energy):
    dry_soil_sh = dry_soil_vol_heat_capacity * depth
    UI3 = (dry_soil_sh + soil_water * DGCM_SPECIFIC_HEAT_ICE) * DGCM_T3
    UL3 = UI3 + soil_water * DGCM_LATENT_HEAT_FUSION_3

    frozen_T = (internal_energy / UI3) * DGCM_T3
    unfrozen_T = ((internal_energy
                   + soil_water * DGCM_SPECIFIC_HEAT_WATER
                   * DGCM_T_LIQUID_H2O_ZERO_ENERGY)
                  / (dry_soil_sh + soil_water * DGCM_SPECIFIC_HEAT_WATER))
    plateau_LF = (internal_energy - UI3) / (soil_water
                                            * DGCM_LATENT_HEAT_FUSION_3)

    is_frozen = internal_energy < UI3
    is_liquid = internal_energy > UL3

    TEMP = jnp.where(is_frozen, frozen_T,
                     jnp.where(is_liquid, unfrozen_T, DGCM_TK0C))
    LF = jnp.where(is_frozen, 0.0,
                   jnp.where(is_liquid, 1.0, plateau_LF))
    return TEMP, LF
