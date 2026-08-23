"""Literals transcribed from C/projects/CARDAMOM_GENERAL/GLOBAL_CONSTANTS.c.

Transcribed EXACTLY — including the 7-digit pi (BUG_COMPAT: pi_7digit).
Never substitute math.pi/np.pi anywhere in dalec_jax: the C model's solar
geometry, daylength, and every downstream state depend on this literal.
"""

DGCM_PI = 3.1415927                    # BUG_COMPAT: pi_7digit
DGCM_T3 = 273.16                       # triple point, K
DGCM_TK0C = 273.15                     # 0 degC in K
DGCM_SPECIFIC_HEAT_ICE = 2093.         # J kg-1 K-1
DGCM_SPECIFIC_HEAT_WATER = 4186.       # J kg-1 K-1
DGCM_LATENT_HEAT_VAPORIZATION = 2.501e6  # J kg-1
DGCM_LATENT_HEAT_FUSION_3 = 3.34e5
DGCM_T_LIQUID_H2O_ZERO_ENERGY = 56.79  # K
DGCM_SEC_DAY = 24 * 60 * 60
