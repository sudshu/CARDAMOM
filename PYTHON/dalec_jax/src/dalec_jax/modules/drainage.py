"""DRAINAGE.c — field-capacity drainage with excess-saturation handling.

Order matters and is preserved from C: sm_field from PSI2MOI first, then the
sm>1 excess split, then delta_sm on the CLAMPED sm, then psi from the clamped
sm. The C `if (sm > 1)` mutation becomes a where-pair; both branches of every
where are value-safe (sm=0 → psi=-inf flows through fmin/fmax exactly as C).
"""
import jax.numpy as jnp

from .hydrofun import hydrofun_moi2psi, hydrofun_psi2moi


def drainage(sm, Qexcess, psi_field, psi_porosity, b):
    sm_field = hydrofun_psi2moi(psi_field, psi_porosity, b)

    over = sm > 1
    excess_drainage = jnp.where(over, sm - 1, 0.0)
    sm = jnp.where(over, 1.0, sm)

    delta_sm = jnp.fmax(sm - sm_field, 0.0)
    # GRADIENT HARDENING (value-identical): for sm <= 0 the C's pow yields
    # ±inf/NaN and the fmin/fmax pair clamps psi to psi_field, making the
    # drainage factor exactly 0 (and delta_sm is 0 there anyway). We feed
    # pow a safe operand and select psi_field for sm <= 0 — same clamped
    # value, finite backward pass.
    psi = hydrofun_moi2psi(jnp.where(sm > 0, sm, 1.0), psi_porosity, b)
    psi = jnp.where(sm > 0, psi, psi_field)

    return excess_drainage + delta_sm * Qexcess * (
        1 - (psi_porosity - jnp.fmin(jnp.fmax(psi, psi_field), psi_porosity))
        / (psi_porosity - psi_field))
