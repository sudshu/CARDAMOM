"""INTERNAL_ENERGY_PER_LIQUID_H2O_UNIT_MASS.c + INITIALIZE_INTERNAL_SOIL_ENERGY.c."""
from ..constants import (DGCM_SPECIFIC_HEAT_WATER,
                         DGCM_T_LIQUID_H2O_ZERO_ENERGY)
from .soil_temp_liquid_frac import soil_temp_and_liquid_frac


def internal_energy_per_liquid_h2o_unit_mass(TEMP):
    # C line 8: U = cw*(TEMP - T_zero_energy); assumes liquid fraction 1
    return DGCM_SPECIFIC_HEAT_WATER * (TEMP - DGCM_T_LIQUID_H2O_ZERO_ENERGY)


def initialize_internal_soil_energy(internal_energy_per_mm_H2O, H2O_mm,
                                    dry_soil_vol_heat_capacity, depth):
    # C: SOIL_TEMP_AND_LIQUID_FRAC on a 1 mm, no-soil column to get the
    # water temperature implied by the per-mm energy, then total energy.
    TEMP, _LF = soil_temp_and_liquid_frac(0.0, 0.0, 1.0,
                                          internal_energy_per_mm_H2O)
    SOIL_E = dry_soil_vol_heat_capacity * depth * TEMP
    H2O_E = internal_energy_per_mm_H2O * H2O_mm
    return SOIL_E + H2O_E
