"""LAI_KNORR_funcs.c — smoothing helpers + daylength for the Knorr phenology.

ComputeDaylightHours uses the 7-digit DGCM_PI (BUG_COMPAT: pi_7digit) — at
latitude 90 the tan() argument is not exactly pi/2, so there is no pole
singularity, exactly as in C.
"""
import jax.numpy as jnp

from ..constants import DGCM_PI


def min_quadratic_smooth(x, y, eta):
    z = jnp.power(x + y, 2) - 4.0 * eta * x * y
    z = jnp.fmax(z, 1e-18)
    return (x + y - jnp.sqrt(z)) / (2.0 * eta)


def max_exponential_smooth(x, y, x0):
    # C branches on x >= (y - x0); the exp argument is finite on the taken
    # branch; on the discarded branch it may overflow to +inf, which where()
    # drops — value-identical to C.
    smooth = x + x0 * jnp.exp(-(x - y) / x0 - 1.0)
    return jnp.where(x >= (y - x0), smooth, y)


def compute_daylight_hours(latitude, DOY):
    dec = (-23.4 * jnp.cos((360. * (DOY + 10.) / 365.) * DGCM_PI / 180.)
           * DGCM_PI / 180.)
    mult = jnp.tan(latitude * DGCM_PI / 180) * jnp.tan(dec)
    dayl = 24. * jnp.arccos(-mult) / DGCM_PI
    return jnp.where(mult >= 1, 24.0, jnp.where(mult <= -1, 0.0, dayl))
