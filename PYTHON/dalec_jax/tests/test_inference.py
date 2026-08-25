"""Contract tests for the inference fast path.

These use cheap synthetic targets rather than the 89-parameter model: the
invariants below are properties of the machinery, not of DALEC, so they
run in milliseconds and still catch the failures that matter. The real
model is exercised end-to-end by examples/laplace_fast_path.py.

The P_end/z_end consistency test is a regression guard: multipoint_laplace
once returned the post-update point paired with the pre-update objective,
which misreported mode quality by up to 11 log-units and could pair a
finite P with a NaN z.
"""
from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy
pytest.importorskip("optax")

from dalec_jax.inference import (cap_covariance, dedupe_modes,  # noqa: E402
                                 evidence_weights, exact_hessians,
                                 multipoint_laplace, run_rwm)

PRIOR_VAR_CAP = np.pi ** 2 / 3


def _banana(z):
    """Smooth, curved, well-behaved 4-D target with a single mode at 0."""
    a = z[0] ** 2 + 3.0 * (z[1] - 0.3 * z[0] ** 2) ** 2
    return -0.5 * (a + jnp.sum(z[2:] ** 2))


def _cliff(z):
    """Same, but -inf outside a box — the hard-EDC pathology in miniature."""
    return jnp.where(jnp.max(jnp.abs(z)) > 2.5, -jnp.inf, _banana(z))


@pytest.mark.parametrize("target", [_banana, _cliff])
def test_laplace_pend_matches_logpost_at_zend(target):
    z0 = np.array([[1.0, 0.5, -0.4, 0.2], [-1.2, 0.9, 0.3, -0.7],
                   [0.4, -1.1, 0.8, 0.1]])
    res = multipoint_laplace(target, z0, max_iters=40, chunk=10,
                             verbose=False)
    assert np.isfinite(res["z_end"]).all(), "returned a non-finite iterate"
    fresh = np.array([float(target(jnp.asarray(z))) for z in res["z_end"]])
    np.testing.assert_allclose(res["P_end"], fresh, rtol=0, atol=1e-12)


def test_laplace_improves_on_the_start():
    z0 = np.array([[1.5, -1.0, 0.7, 0.9], [-0.8, 1.4, -0.6, 0.5]])
    start = np.array([float(_banana(jnp.asarray(z))) for z in z0])
    res = multipoint_laplace(_banana, z0, max_iters=60, chunk=10,
                             verbose=False)
    assert (res["P_end"] >= start - 1e-9).all()
    assert res["P_end"].max() > -1e-3          # mode of _banana is 0.0


def test_cap_covariance_caps_flat_directions_at_the_prior_width():
    # exact_hessians/cap_covariance work with the Hessian of -logpost, so
    # curvature is positive: 1e-12 is likelihood-flat, 4.0 is well-informed
    H = np.diag([1e-12, 4.0, 1.0])
    cov = cap_covariance(H)
    ev = np.sort(np.linalg.eigvalsh(cov))
    assert ev[-1] == pytest.approx(PRIOR_VAR_CAP), "flat direction uncapped"
    assert ev[0] == pytest.approx(0.25)        # 1/4, left alone
    assert np.allclose(cov, cov.T)


def test_dedupe_modes_drops_nonfinite_and_duplicates():
    z = np.array([[0.0, 0.0], [0.0005, -0.0005], [3.0, 3.0], [1.0, 1.0]])
    P = np.array([-1.0, -2.0, -0.5, np.nan])
    keep = dedupe_modes(z, P, tol=0.15)
    assert 3 not in keep, "kept a NaN-valued mode"
    assert 1 not in keep, "kept a duplicate of mode 0"
    assert set(keep) == {0, 2}
    assert keep[0] == 2, "modes should be ordered best-P first"


def test_exact_hessian_matches_analytic():
    quad = lambda z: -0.5 * (2.0 * z[0] ** 2 + 5.0 * z[1] ** 2)
    # exact_hessians returns the Hessian of -logpost (positive definite
    # at a mode), which is what cap_covariance consumes
    H = exact_hessians(quad, np.zeros((1, 2)), verbose=False)[0]
    np.testing.assert_allclose(H, np.diag([2.0, 5.0]), atol=1e-9)


def test_evidence_weights_normalize_and_favour_the_better_mode():
    covs = np.stack([np.eye(2), np.eye(2)])
    w = evidence_weights(np.array([-1.0, -3.0]), covs)
    assert w.sum() == pytest.approx(1.0)
    assert w[0] > w[1]


def test_evidence_weights_survive_one_unusable_mode():
    """A non-finite Hessian at one cliff-side mode must not poison the rest.

    Regression: a single NaN logdet made logw.max() NaN and therefore
    every weight NaN, which downstream degraded silently to "argmax picks
    mode 0".
    """
    covs = np.stack([np.eye(2), np.full((2, 2), np.nan), np.eye(2)])
    w = evidence_weights(np.array([-1.0, -0.5, -3.0]), covs)
    assert np.isfinite(w).all()
    assert w.sum() == pytest.approx(1.0)
    assert w[1] == 0.0, "unusable mode must get zero weight"
    assert w[0] > w[2], "ranking among usable modes preserved"


def test_evidence_weights_reject_non_positive_definite_covariance():
    covs = np.stack([np.eye(2), np.diag([1.0, -1.0])])
    w = evidence_weights(np.array([-1.0, -0.1]), covs)
    assert w[1] == 0.0 and w[0] == pytest.approx(1.0)


def test_evidence_weights_fall_back_to_uniform_if_nothing_is_usable():
    covs = np.stack([np.full((2, 2), np.nan), np.full((2, 2), np.nan)])
    w = evidence_weights(np.array([-1.0, -2.0]), covs)
    assert w.sum() == pytest.approx(1.0)
    assert np.allclose(w, 0.5)


def test_rwm_shapes_and_acceptance():
    z0 = np.zeros((4, 4))
    out = run_rwm(_banana, z0, np.eye(4), n_iters=200, chunk=100, thin=10,
                  scale=0.5, seed=3, verbose=False)
    assert out["z"].shape == (4, 20, 4)
    assert 0.0 < out["acc"] <= 1.0
    assert np.isfinite(out["z"]).all()


def test_rwm_never_leaves_the_feasible_region():
    """A chain started inside a hard boundary must stay inside it."""
    z0 = np.zeros((8, 4))
    out = run_rwm(_cliff, z0, np.eye(4), n_iters=400, chunk=200, thin=10,
                  scale=0.8, seed=5, verbose=False)
    assert np.abs(out["z"]).max() <= 2.5


def test_rwm_rejects_a_non_finite_proposal_covariance():
    """Regression: a NaN covariance froze chains at 0% acceptance silently.

    Cholesky of NaN gives NaN, every proposal is NaN, every Metropolis
    ratio compares False, and the run looks like a merely hard target.
    Observed on FluxVal site 71 (chain movement exactly 0.0).
    """
    cov = np.eye(4)
    cov[2, 2] = np.nan
    with pytest.raises(ValueError, match="non-finite"):
        run_rwm(_banana, np.zeros((2, 4)), cov, n_iters=100, chunk=100,
                thin=10, seed=1, verbose=False)


def test_rwm_is_deterministic_for_a_fixed_seed():
    z0 = np.zeros((2, 4))
    kw = dict(n_iters=200, chunk=100, thin=10, scale=0.5, verbose=False)
    a = run_rwm(_banana, z0, np.eye(4), seed=11, **kw)
    b = run_rwm(_banana, z0, np.eye(4), seed=11, **kw)
    np.testing.assert_array_equal(a["z"], b["z"])
    assert a["acc"] == b["acc"]
