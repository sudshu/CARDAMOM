"""The sampling target: z-space log-posterior identical to the C sampler's.

The C MHMCMC works in normalized u in [0,1]^89 with a UNIFORM prior and the
log-ratio mapping p = pmin * (pmax/pmin)**u (NORMPARS.c). We sample in
z = logit(u), so

    log pi(z) = P(p(u(z))) + sum_k [ log sigmoid(z_k) + log sigmoid(-z_k) ]

where the second term is the logit-transform Jacobian. The pushforward to
u-space therefore matches the C target exactly.

CAUTION (learned the hard way): the value returned by the log-posterior is
the *z-space density*, which includes the Jacobian term (typically ~ -158
at posterior-typical points here). Never compare it directly against the
C's model log-posterior for chain samples — subtract
:func:`logit_jacobian` first, or evaluate the raw model P separately.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from .. import edcs
from ..indices import PARMAX, PARMIN
from ..likelihood import data_prep, mlf2
from ..model import prederive_vegk, run_dalec_1100


def nor2par(u):
    """Normalized u in [0,1]^89 -> physical parameters (NORMPARS.c mapping)."""
    pmin = jnp.asarray(PARMIN)
    return pmin * jnp.exp(u * jnp.log(jnp.asarray(PARMAX) / pmin))


def par2nor(p):
    """Physical parameters -> normalized u in [0,1]^89."""
    pmin = jnp.asarray(PARMIN)
    return jnp.log(p / pmin) / jnp.log(jnp.asarray(PARMAX) / pmin)


def logit_jacobian(z):
    """The transform term included in build_logpost's return value."""
    return jnp.sum(jax.nn.log_sigmoid(z) + jax.nn.log_sigmoid(-z))


def build_logpost(cbf_path: str, gate: str = "hard"):
    """Return (logpost, cbf): logpost(z) -> z-space log-posterior scalar.

    logpost is a pure JAX function of z (shape (89,)); vmap/grad/jit at
    will. It evaluates the full pipeline: 240-step forward model, all 15
    EDCs, the 31-term MLF2 likelihood, plus the logit Jacobian.

    gate:
      "hard" (default) — the C-exact target: -inf outside the EDC-feasible
        set. The only gate mode for sampling and for anything published.
      "none" — the likelihood WITHOUT the EDC gate. Inside the feasible
        set this is numerically IDENTICAL to "hard" (the gate is
        -inf-or-zero, never a slope), but it is differentiable everywhere:
        derivatives no longer pass through the -inf `where` branches that
        make Hessians non-finite at cliff-adjacent points (measured at
        NL-Loo: 10/24 high-P chain draws). Use for curvature (Laplace /
        atlas / Newton) and ONLY at points verified feasible by the hard
        gate; the pushforward density it defines outside the feasible set
        is not the posterior.
    """
    if gate not in ("hard", "none"):
        raise ValueError(f"gate must be 'hard' or 'none', got {gate!r}")
    cbf = data_prep.load_cbf(cbf_path)
    ecfg = {"n_timesteps": cbf.n_timesteps,
            "dint": edcs.compute_dint(cbf.time),
            "edc_eqf": cbf.edc_eqf, "skt_ref_mean": cbf.skt_ref_mean}
    VegK = prederive_vegk(cbf.met["DOY"], cbf.LAT)
    pmin = jnp.asarray(PARMIN)
    lratio = jnp.log(jnp.asarray(PARMAX) / pmin)

    if gate == "hard":
        def logpost(z):
            u = jax.nn.sigmoid(z)
            p = pmin * jnp.exp(u * lratio)
            pools, fluxes = run_dalec_1100(p, cbf.met, cbf.LAT,
                                           cbf.deltat, VegK)
            _, _, P = mlf2(cbf, ecfg, p, pools, fluxes)
            return P + logit_jacobian(z)
    else:
        from ..likelihood import likelihood

        def logpost(z):
            u = jax.nn.sigmoid(z)
            p = pmin * jnp.exp(u * lratio)
            pools, fluxes = run_dalec_1100(p, cbf.met, cbf.LAT,
                                           cbf.deltat, VegK)
            _, Plik = likelihood(cbf, p, pools, fluxes)
            return Plik + logit_jacobian(z)

    return logpost, cbf
