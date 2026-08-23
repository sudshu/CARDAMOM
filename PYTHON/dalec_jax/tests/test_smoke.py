"""P0 smoke: the environment can express the constructs the port relies on.

Checks x64 end-to-end through the exact primitive combination the model uses
(lax.scan carry + jnp.where selects + vmap over parameter vectors), and that
the erfc/exp/pow primitives the census will characterize are float64.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import dalec_jax  # noqa: F401  (enables + verifies x64)


def test_x64_active():
    assert jnp.zeros(3).dtype == jnp.float64
    assert jnp.array(3.1415927).dtype == jnp.float64


def test_scan_where_vmap_pipeline():
    # miniature of the model shape: carry=pools, per-step where-clamped update
    def step(pools, met):
        upd = pools * jnp.exp(-0.01 * met) + 0.1
        upd = jnp.where(upd > 0.05, upd, 0.05)  # clamp idiom
        return upd, upd.sum()

    met = jnp.linspace(0.0, 1.0, 240)
    p0 = jnp.full((14,), 100.0)

    def run(scale):
        final, series = jax.lax.scan(step, p0 * scale, met)
        return series

    series = jax.vmap(run)(jnp.array([0.5, 1.0, 2.0]))
    assert series.shape == (3, 240)
    assert series.dtype == jnp.float64
    assert bool(jnp.isfinite(series).all())


def test_alive_flag_freeze_to_zero():
    # the isfinite-break replacement: poisoned step recorded, zeros after
    def step(carry, x):
        pools, alive = carry
        nxt = jnp.where(alive, pools - x, pools)
        nxt_alive = jnp.logical_and(alive, jnp.isfinite(nxt).all())
        out = jnp.where(alive, nxt, jnp.zeros_like(nxt))  # C calloc-zero tail
        return (out, nxt_alive), out

    xs = jnp.array([1.0, jnp.inf, 1.0, 1.0])
    (_, alive), traj = jax.lax.scan(step, (jnp.ones(2), jnp.array(True)), xs)
    assert not bool(alive)
    # step 0 ran; step 1 records the poisoned (inf) values; later steps zero
    np.testing.assert_array_equal(np.asarray(traj[0]), [0.0, 0.0])
    assert not np.isfinite(np.asarray(traj[1])).any()
    np.testing.assert_array_equal(np.asarray(traj[2:]), 0.0)


def test_transcendentals_are_f64():
    x = jnp.array([0.3, 1.7])
    for y in (jnp.exp(x), jnp.log(x), jnp.power(x, 2.5),
              jax.scipy.special.erfc(x), jnp.sqrt(x)):
        assert y.dtype == jnp.float64


@pytest.mark.skipif(not any(d.platform == "gpu" for d in jax.devices()),
                    reason="no GPU visible")
def test_gpu_x64_scan():
    gpu = next(d for d in jax.devices() if d.platform == "gpu")
    x = jax.device_put(jnp.ones(16), gpu)
    y, _ = jax.lax.scan(lambda c, _: (c * 1.0000001, c.sum()), x, None, length=100)
    assert y.dtype == jnp.float64
