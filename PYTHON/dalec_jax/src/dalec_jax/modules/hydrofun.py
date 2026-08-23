"""HYDROLOGY_MODULES/CONVERTERS — 1:1 ports of the five converters.

C sources: HYDROFUN_{EWT2MOI,MOI2EWT,MOI2CON,MOI2PSI,PSI2MOI}.c.
No clamps beyond the C's own: MOI2PSI(0) follows IEEE through 1/0 = inf,
pow(inf, b) = inf, psi = -inf — the old in-repo port's max(moi, 1e-4) clamp
is exactly the kind of divergence this package forbids.
"""
import jax.numpy as jnp


def hydrofun_ewt2moi(ewt, p, z):
    # HYDROFUN_EWT2MOI.c: moi = ewt/(1000*p*z)
    return ewt / (1000 * p * z)


def hydrofun_moi2ewt(moi, p, z):
    # HYDROFUN_MOI2EWT.c: ewt = moi*1000*p*z
    return moi * 1000 * p * z


def hydrofun_moi2con(moi, k0, b):
    # HYDROFUN_MOI2CON.c: con = k0*pow(moi, 2*b+3)
    return k0 * jnp.power(moi, 2 * b + 3)


def hydrofun_moi2psi(moi, psi_porosity, b):
    # HYDROFUN_MOI2PSI.c: psi = psi_porosity*(pow((1/moi), b))
    return psi_porosity * jnp.power(1 / moi, b)


def hydrofun_psi2moi(psi, psi_porosity, b):
    # HYDROFUN_PSI2MOI.c: moi = pow(psi/psi_porosity, (-1/b))
    return jnp.power(psi / psi_porosity, -1 / b)
