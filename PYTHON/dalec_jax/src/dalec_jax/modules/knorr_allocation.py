"""KNORR_ALLOCATION.c — Knorr et al. (2010) leaf phenology, updated per
Norton et al. (2023). erfc is the package's sole transcendental that is not
bit-identical to glibc (TOLERANCES.md census: <=34 ULP, 4.1e-15 rel).
"""
import jax.numpy as jnp
import jax.scipy.special as jsp

from .lai_knorr_funcs import (compute_daylight_hours, max_exponential_smooth,
                              min_quadratic_smooth)


def knorr_allocation(temp, deltat, n, latitude, DOY, lam, lambda_max,
                     T_phi, T_r, plgr, k_L, pasm, transp, tau_W,
                     t_c, t_r, T_memory, lambda_max_memory):
    tau_m = 30.0
    tau_s = 30.0

    T = jnp.exp(-deltat / tau_m) * T_memory + temp * (1 - jnp.exp(-deltat / tau_m))
    T_deviation = (T - T_phi) / T_r
    f_T = 0.5 * jsp.erfc(-T_deviation * jnp.sqrt(0.5))

    daylength = compute_daylight_hours(latitude, DOY)
    td_deviation = (daylength - t_c) / t_r
    f_d = 0.5 * jsp.erfc(-td_deviation * jnp.sqrt(0.5))

    f = f_T * f_d
    r = plgr * f + (1 - f) * k_L

    lambda_W = (pasm * lam) / (tau_W * max_exponential_smooth(transp, 1e-3, 2e-2))
    lambda_tilde_max = min_quadratic_smooth(lambda_max, lambda_W, 0.99)
    laim = (jnp.exp(-deltat / tau_s) * lambda_max_memory
            + lambda_tilde_max * (1.0 - jnp.exp(-deltat / tau_s)))
    lambda_lim = max_exponential_smooth(plgr * laim * f / r, 1e-9, 5e-3)

    lambda_next = lambda_lim - (lambda_lim - lam) * jnp.exp(-r * deltat)
    dlambdadt = lambda_next - lam

    return (lambda_next, T, laim, dlambdadt, f_T, f_d,
            lambda_tilde_max, lambda_W)
