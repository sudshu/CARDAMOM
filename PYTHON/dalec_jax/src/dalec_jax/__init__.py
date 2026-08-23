"""dalec_jax: JAX port of CARDAMOM's DALEC_1100 with C-verified equivalence.

Float64 is a correctness requirement, not a preference (see CLAUDE.md and
BUG_COMPAT `lf_exact_eq`): x64 is enabled at import, before any jax array is
created, and verified.
"""
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp

if jnp.zeros(1).dtype != jnp.float64:
    raise RuntimeError(
        "dalec_jax requires float64; jax_enable_x64 failed to take effect "
        "(was jax imported and used before dalec_jax?)"
    )

__version__ = "0.1.0"
