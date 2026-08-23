"""1:1 JAX ports of the DALEC_1100 C leaf modules.

ORACLE_REGISTRY maps each oracle module name (see `oracle_1100 manifest`) to
a python function taking the manifest's inputs positionally and returning the
manifest's outputs as a tuple (single outputs unwrapped by the caller).
"""
from .alloc_and_auto_resp import alloc_and_auto_resp_fluxes
from .drainage import drainage
from .het_resp_rates_jcr import het_resp_rates_jcr
from .hydrofun import (hydrofun_ewt2moi, hydrofun_moi2con, hydrofun_moi2ewt,
                       hydrofun_moi2psi, hydrofun_psi2moi)
from .knorr_allocation import knorr_allocation
from .lai_knorr_funcs import (compute_daylight_hours, max_exponential_smooth,
                              min_quadratic_smooth)
from .liu_an_et import liu_an_et
from .soil_energy import (initialize_internal_soil_energy,
                          internal_energy_per_liquid_h2o_unit_mass)
from .soil_temp_liquid_frac import soil_temp_and_liquid_frac

ORACLE_REGISTRY = {
    "HYDROFUN_EWT2MOI": hydrofun_ewt2moi,
    "HYDROFUN_MOI2EWT": hydrofun_moi2ewt,
    "HYDROFUN_MOI2CON": hydrofun_moi2con,
    "HYDROFUN_MOI2PSI": hydrofun_moi2psi,
    "HYDROFUN_PSI2MOI": hydrofun_psi2moi,
    "DRAINAGE": drainage,
    "INTERNAL_ENERGY_PER_LIQUID_H2O_UNIT_MASS":
        internal_energy_per_liquid_h2o_unit_mass,
    "INITIALIZE_INTERNAL_SOIL_ENERGY": initialize_internal_soil_energy,
    "MIN_QUADRATIC_SMOOTH": min_quadratic_smooth,
    "MAX_EXPONENTIAL_SMOOTH": max_exponential_smooth,
    "COMPUTE_DAYLIGHT_HOURS": compute_daylight_hours,
    "SOIL_TEMP_AND_LIQUID_FRAC": soil_temp_and_liquid_frac,
    "HET_RESP_RATES_JCR": het_resp_rates_jcr,
    "KNORR_ALLOCATION": knorr_allocation,
    "ALLOC_AND_AUTO_RESP_FLUXES": alloc_and_auto_resp_fluxes,
    "LIU_AN_ET": liu_an_et,
}
