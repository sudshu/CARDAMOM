"""Regression tests for the C/JAX trajectory verdict (dalec_jax.verification).

The defect these lock down: `laplace_modes[1]` from the BR-Sa1 mechanistic
ensemble was reported as a *genuine* C/JAX discrepancy on two forcing paths in
a 56-vector batch, and as *chaos-certified* on one of those paths when the same
vector was run alone. The verdict has to be a function of the vector and the
forcing path — nothing else.

What actually diverges on both paths is flux 87 `nonleaf_mortality_factor`,
0.0 in the C against 1.0 in JAX. The stand has died; the C's live carbon pools
land on exactly +0.0 through cancellation in an absorbing pool update while
JAX's land on small non-zero values, and
`ALLOC_AND_AUTO_RESP_FLUXES.c:65` guards on
`POTENTIAL_AUTO_RESP_MAINTENANCE == 0`. (No underflow is involved: the C goes
-8.3e-26 -> +0.0 in one step, and every value on the JAX side is a normal
float64.) The two tolerance-shaped red herrings in the original report — that
the difference is 2 ULP on a pool at 1.4e9, and that 1.4e9 is a physically
absurd carbon pool — are both wrong: pool 11 is E_LY1, thermal energy in J m-2.
"""
from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

import dalec_jax  # noqa: F401  (enables x64 before anything builds an array)
from dalec_jax import verification as V
from dalec_jax.indices import F, NOFLUXES, NOPOOLS, S

PKG = Path(__file__).resolve().parents[1]
DATA = PKG / "tests/data"
ORACLE = PKG.parents[1] / "C/projects/JAX_VALIDATION/oracle_1100.exe"
SEED = 20260828

DONOR_CBFS = {
    "2004-07": DATA / "chaos_D_donor_2004-07.cbf.nc",
    "2006-07": DATA / "chaos_D_donor_2006-07.cbf.nc",
}

#: 55 rows of the tracked `tests/data/assim_1100.cbr` posterior that run to
#: completion on BOTH donor paths — no early break, no C/JAX divergence, final
#: live carbon 5.3e2..1.1e4 gC m-2. Selected offline by running the first 1600
#: rows through both engines (1088 of 1600 qualify). The near-degenerate
#: `viable_ensemble` members all break within 121 steps on these paths and
#: exercise almost no vectorised codegen, which is why they are not used here.
HEALTHY_ROWS = (
    1, 99, 152, 162, 186, 202, 254, 265, 299, 342, 360, 386, 400, 401, 402,
    403, 404, 405, 406, 407, 408, 410, 411, 412, 413, 414, 415, 416, 417,
    418, 419, 420, 422, 423, 424, 426, 427, 428, 429, 430, 431, 432, 433,
    434, 435, 436, 438, 439, 440, 441, 442, 443, 444, 445, 446,
)


# ----------------------------------------------------- the element gate


def test_element_criterion_is_the_locked_l4_contract():
    """The shared gate must be the contract, verbatim — never a loosening."""
    rng = np.random.default_rng(0)
    c = rng.normal(size=(40, 7)) * 10.0 ** rng.integers(-14, 10, size=(40, 7))
    j = c * (1 + rng.normal(scale=1e-9, size=c.shape))
    c[3, 2] = np.inf
    j[3, 2] = np.inf
    c[5, 4] = np.nan
    j[5, 4] = 1.0

    fin = np.isfinite(c)
    with np.errstate(all="ignore"):
        rms = np.nan_to_num(
            np.sqrt(np.nanmean(np.where(fin, c, np.nan) ** 2, axis=0)), nan=1.0)
        mixed = np.abs(j - c) / np.maximum(
            np.abs(np.where(fin, c, 0.0)), np.maximum(rms, 1e-300))
    ok = (mixed <= 1e-10) | (np.abs(j - c) <= 1e-12)
    expected = np.where(fin & np.isfinite(j), ~ok,
                        ~((~np.isfinite(c)) & (~np.isfinite(j))))

    assert np.array_equal(V.element_fail(c, j), expected)
    assert (V.REL_TOL, V.ABS_TOL) == (1e-10, 1e-12)


def test_nonfinite_pairs_pass_as_the_docstring_says():
    """The inherited criterion treats all non-finites as one class; say so."""
    c = np.array([[np.inf, np.nan, np.inf, 1.0]])
    j = np.array([[np.nan, np.inf, -np.inf, 1.0]])
    assert not V.element_fail(c, j).any()
    # only a finite/non-finite mismatch fails
    assert V.element_fail(np.array([[np.inf]]), np.array([[1.0]]))[0, 0]


def test_divergence_step_rejects_a_mismatched_flux_length():
    cp = np.zeros((5, NOPOOLS))
    cf = np.zeros((5, NOFLUXES))          # should be 4
    with pytest.raises(ValueError, match="one fewer flux row"):
        V.divergence_step(cp, cp.copy(), cf, cf.copy())


# ------------------------------------------- batch-position independence


def _legacy_dither_block(params_2d, k, seed):
    """chaos_onsets() as it was: ONE stream shaped (n, K, nopars).

    Row-major fill means a vector's directions depend on its ordinal position
    in the block, so the same vector gets different dithers depending on which
    other vectors happened to land in the block with it.
    """
    params_2d = np.asarray(params_2d, dtype=np.float64)
    rng = np.random.default_rng(seed)
    dith = np.repeat(params_2d[:, None, :], k, axis=1)
    up = rng.random(dith.shape) < 0.5
    return np.where(up, np.nextafter(dith, np.inf),
                    np.nextafter(dith, -np.inf))


def test_dither_block_does_not_depend_on_batch_position():
    rng = np.random.default_rng(7)
    target = rng.normal(size=89)
    others = rng.normal(size=(6, 89))

    alone = V.dither_block(target, V.DITHER_K, SEED)
    for pos in (0, 1, 3, 6):
        block = np.vstack([others[:pos], target[None], others[pos:]])
        in_batch = V.dither_block(block[pos], V.DITHER_K, SEED)
        assert np.array_equal(in_batch, alone), \
            f"dithers changed when the vector sat at position {pos}"

    # ...and the construction this replaces really did move with position,
    # so the test above is not vacuous.
    legacy_alone = _legacy_dither_block(target[None], V.DITHER_K, SEED)[0]
    legacy_at_1 = _legacy_dither_block(
        np.vstack([others[:1], target[None]]), V.DITHER_K, SEED)[1]
    assert not np.array_equal(legacy_alone, legacy_at_1)


def test_dither_block_is_vector_specific_and_reproducible():
    a = np.arange(89, dtype=float) + 1.0
    b = a.copy()
    b[17] = np.nextafter(b[17], np.inf)
    assert np.array_equal(V.dither_block(a, 8, SEED),
                          V.dither_block(a.copy(), 8, SEED))
    assert not np.array_equal(V.dither_block(a, 8, SEED),
                              V.dither_block(b, 8, SEED))
    assert not np.array_equal(V.dither_block(a, 8, SEED),
                              V.dither_block(a, 8, SEED + 1))
    # every dither is exactly 1 ULP away in every coordinate
    d = V.dither_block(a, 8, SEED)
    up = np.nextafter(a, np.inf)
    dn = np.nextafter(a, -np.inf)
    assert np.all((d == up) | (d == dn))


def test_dither_escalation_only_extends_the_block():
    a = np.arange(89, dtype=float) + 1.0
    small = V.dither_block(a, V.DITHER_K, SEED)
    big = V.dither_block(a, V.DITHER_K_ESCALATED, SEED)
    assert np.array_equal(big[:V.DITHER_K], small)


# ----------------------------------------------- the magnitude/ULP report


def _synthetic_pair(t=8):
    """C/JAX blocks reproducing the real signature of this bug."""
    cp = np.zeros((t + 1, NOPOOLS))
    cp[:, 11] = 1.4089653908132904e9          # E_LY1, J m-2 — normal
    cp[:, S.C_woo] = 500.0
    jp = cp.copy()
    jp[5, 11] = np.nextafter(np.nextafter(cp[5, 11], np.inf), np.inf)
    cf = np.zeros((t, NOFLUXES))
    cf[:, F.nonleaf_mortality_factor] = [0.9, 0.9, 0.9, 0.0, 0.0, 0.0, 0.0, 0.0]
    jf = cf.copy()
    jf[5, F.nonleaf_mortality_factor] = 1.0
    return cp, jp, cf, jf


def test_report_names_the_element_that_failed_not_a_drifting_pool():
    cp, jp, cf, jf = _synthetic_pair()

    # the 2-ULP pool drift at 1.4e9 is NOT a divergence under the contract
    assert not V.element_fail(cp, jp)[5, 11]

    d = V.first_divergence(cp, jp, cf, jf)
    assert d is not None
    assert (d.kind, d.index, d.name) == (
        "flux", F.nonleaf_mortality_factor, "nonleaf_mortality_factor")
    assert d.step == 5
    assert (d.c_value, d.jax_value) == (0.0, 1.0)
    assert d.abs_diff == 1.0
    assert d.rel_to_scale > 1.0            # full-scale, not last-bit
    assert d.ulps > 10 ** 15               # 0.0 vs 1.0 is not a near miss
    assert V.divergence_step(cp, jp, cf, jf) == 5


def test_a_nan_element_never_outranks_the_real_failure():
    """A NaN ratio must not win "worst element" by iteration order."""
    cp, jp, cf, jf = _synthetic_pair()
    cp[5, S.C_lab] = np.nan                # C non-finite, JAX finite -> fails
    jp[5, S.C_lab] = 1.0
    els = V.divergent_elements(cp, jp, cf, jf, 5)
    assert any(math.isnan(e.rel_to_scale) for e in els), "no NaN ratio present"
    for order in (els, list(reversed(els))):
        assert max(order, key=V._severity).name == "nonleaf_mortality_factor"
    assert V.first_divergence(cp, jp, cf, jf).name == "nonleaf_mortality_factor"


def test_a_last_bit_difference_at_any_magnitude_is_not_a_divergence():
    """The criterion is scale-relative at both ends of the dynamic range."""
    cf = np.zeros((3, NOFLUXES))
    for magnitude in (1e-14, 1.0, 1.4089653908132904e9):
        cp = np.zeros((4, NOPOOLS))
        cp[:, 0] = magnitude
        jp = cp.copy()
        jp[2, 0] = np.nextafter(np.nextafter(cp[2, 0], np.inf), np.inf)
        assert V.first_divergence(cp, jp, cf, cf.copy()) is None, \
            f"2 ULP at {magnitude:g} should not be a divergence"


# ------------------------------------------------- the plausibility axis


def test_plausibility_is_separate_and_never_softens_the_verdict():
    cp, jp, cf, jf = _synthetic_pair()
    cp[:, S.C_woo] = 1e-30                 # numerically dead stand
    jp = cp.copy()
    jp[5, 11] = np.nextafter(np.nextafter(cp[5, 11], np.inf), np.inf)

    p = V.state_plausibility(cp, 5)
    assert p.plausible is False
    assert "numerically dead" in p.reason

    # the C reproduces nothing under dither -> the agreement axis still says
    # DISCREPANT; implausibility is recorded beside it, not folded into it
    verdict = V.adjudicate(
        np.arange(89, dtype=float), cp, cf, jp, jf,
        lambda blk: (np.repeat(cp[None], len(blk), 0),
                     np.repeat(cf[None], len(blk), 0)),
        seed=SEED)
    assert verdict.agreement == V.DISCREPANT
    assert verdict.blocks_interpretation is True
    assert verdict.plausibility.plausible is False
    assert verdict.divergence.name == "nonleaf_mortality_factor"

    # a live stand is plausible
    cp[:, S.C_woo] = 500.0
    assert V.state_plausibility(cp, 5).plausible is True


def test_state_plausibility_survives_an_empty_pool_block():
    p = V.state_plausibility(np.zeros((0, NOPOOLS)), 3)
    assert p.plausible is False and "no pool rows" in p.reason


def test_verdict_as_dict_is_strictly_json_serialisable():
    """Reports must survive a strict parser when a non-finite reaches them.

    Reachable per CLAUDE.md rule 8: a JAX value that is non-finite where the C
    is finite gives ``abs_diff = inf``, and NaN pools give a NaN
    ``live_carbon``.
    """
    cp = np.ones((4, NOPOOLS))
    cf = np.ones((3, NOFLUXES))
    jp = cp.copy()
    jp[2, 5] = np.inf                      # finite C vs non-finite JAX -> fails
    cp[2, S.C_lab] = jp[2, S.C_lab] = np.nan   # agrees, but NaN live carbon
    v = V.adjudicate(np.arange(89, dtype=float), cp, cf, jp, cf.copy(),
                     lambda blk: (np.repeat(cp[None], len(blk), 0),
                                  np.repeat(cf[None], len(blk), 0)),
                     seed=SEED)
    # NaN ranks last, so the reported element is the infinite one, not C_lab
    assert v.divergence.index == 5
    assert math.isinf(v.divergence.abs_diff)               # raw value kept
    assert math.isinf(v.divergence.rel_to_scale)
    assert math.isnan(v.plausibility.live_carbon)
    d = v.as_dict()
    assert d["divergence"]["abs_diff"] is None             # report is sanitised
    assert d["divergence"]["rel_to_scale"] is None
    assert d["plausibility"]["live_carbon"] is None
    json.dumps(d, allow_nan=False)                         # would raise if not


# ----------------------------------- adjudication, with a stubbed oracle


def _stub_runners(flip_from: int | None, jax_mode: str = "same"):
    """C and JAX stand-ins with the real bug's shape but no model.

    ``flip_from`` is the first dither index at which the stubbed C reproduces
    the branch flip, so a test can put the evidence out of reach of the K=8
    block and inside the K=64 one; ``None`` means the C never reproduces it.

    ``jax_mode`` controls how the stubbed JAX depends on batch width:
    ``"same"`` — width-independent; ``"batch_dirtier"`` — the batch run
    diverges earlier than the single run; ``"batch_cleaner"`` — the batch run
    is CLEAN while the single run diverges, the direction the first version of
    this fix could not see.
    """
    cp, jp, cf, jf = _synthetic_pair()

    def run_c(block):
        block = np.asarray(block, dtype=np.float64)
        pools = np.repeat(cp[None], len(block), 0)
        fluxes = np.repeat(cf[None], len(block), 0)
        if flip_from is not None:
            fluxes[flip_from:, 5, F.nonleaf_mortality_factor] = 1.0
        return pools, fluxes

    def run_jax(block):
        block = np.asarray(block, dtype=np.float64)
        batched = len(block) > 1
        pools = np.repeat(jp[None], len(block), 0)
        fluxes = np.repeat(jf[None], len(block), 0)
        if jax_mode == "batch_dirtier" and batched:
            fluxes[:, 4, F.nonleaf_mortality_factor] = 1.0
        elif jax_mode == "batch_cleaner" and batched:
            fluxes[:] = np.repeat(cf[None], len(block), 0)   # matches the C
        return pools, fluxes

    return run_c, run_jax


def test_adjudicate_escalates_before_calling_a_discrepancy_genuine():
    target = np.arange(89, dtype=float) + 1.0
    cp, jp, cf, jf = _synthetic_pair()

    # the C reproduces the flip inside the K=8 block -> certified straight away
    run_c, _ = _stub_runners(flip_from=0)
    v = V.adjudicate(target, cp, cf, jp, jf, run_c, seed=SEED)
    assert v.agreement == V.CHAOS_CERTIFIED
    assert v.n_dithers == V.DITHER_K

    # the evidence sits beyond K=8. Escalation is monotone and one-sided: more
    # dithers can only turn DISCREPANT into CHAOS_CERTIFIED, never the reverse.
    run_c_late, _ = _stub_runners(flip_from=V.DITHER_K + 4)
    v1 = V.adjudicate(target, cp, cf, jp, jf, run_c_late, seed=SEED)
    assert v1.agreement == V.CHAOS_CERTIFIED
    assert v1.n_dithers == V.DITHER_K_ESCALATED
    assert any("escalated" in n for n in v1.notes)
    assert V.adjudicate(target, cp, cf, jp, jf, run_c_late, seed=SEED,
                        k_escalated=V.DITHER_K).agreement == V.DISCREPANT

    # the C never reproduces it -> genuine, but only after escalating
    run_c_never, _ = _stub_runners(flip_from=None)
    v2 = V.adjudicate(target, cp, cf, jp, jf, run_c_never, seed=SEED)
    assert v2.agreement == V.DISCREPANT
    assert v2.n_dithers == V.DITHER_K_ESCALATED, \
        "a genuine verdict must not be issued before the dither escalation"


def test_a_dither_that_never_diverges_cannot_certify():
    """The fail-closed guard: no self-divergence, no certificate."""
    target = np.arange(89, dtype=float) + 1.0
    cp, jp, cf, jf = _synthetic_pair()
    run_c_never, _ = _stub_runners(flip_from=None)
    v = V.adjudicate(target, cp, cf, jp, jf, run_c_never, seed=SEED)
    assert v.c_self_onset == -1
    assert v.n_dithers_diverged == 0
    assert v.agreement == V.DISCREPANT
    assert v.blocks_interpretation is True


def test_a_nonfinite_dither_certifies_only_via_a_measured_onset():
    """CLAUDE.md rule 8 makes non-finite trajectory values ordinary.

    A dither whose trajectory goes non-finite where the base C is finite HAS
    genuinely left the base trajectory under the locked element criterion, so
    it certifies at that step. That is the locked onset rule reporting a real
    C self-divergence, not a fabricated one, and it is unchanged from the
    harness this replaces (`gen_fixtures.py` scores dithers the same way). It
    is recorded in tests/TOLERANCES.md, because it means certificates on
    near-death vectors rest on the C's own instability.

    What must never happen is a certificate derived from a magnitude envelope
    with a non-finite promoted to ``inf``; that route is not in the module.
    """
    target = np.arange(89, dtype=float) + 1.0
    cp, jp, cf, jf = _synthetic_pair()

    def run_c_nan(block):
        pools = np.repeat(cp[None], len(block), 0)
        fluxes = np.repeat(cf[None], len(block), 0)
        fluxes[0, 5, F.nonleaf_mortality_factor] = np.nan
        return pools, fluxes

    v = V.adjudicate(target, cp, cf, jp, jf, run_c_nan, seed=SEED)
    assert v.agreement == V.CHAOS_CERTIFIED
    assert v.c_self_onset == 5, "certificate must trace to a measured onset"
    assert v.n_dithers_diverged == 1
    assert not hasattr(V, "envelope_covers"), \
        "the magnitude-envelope certification route must stay removed"


@pytest.mark.parametrize("mode", ["batch_dirtier", "batch_cleaner"])
def test_adjudicate_block_verdict_is_invariant_to_batch_width(mode):
    """Both directions: the batch run may be dirtier OR cleaner than alone."""
    rng = np.random.default_rng(3)
    target = np.arange(89, dtype=float) + 1.0
    others = rng.normal(size=(3, 89))
    run_c, run_jax = _stub_runners(flip_from=None, jax_mode=mode)

    alone = V.adjudicate_block(target[None], run_c, run_jax, seed=SEED)[0]
    for pos in (0, 2, 3):
        block = np.vstack([others[:pos], target[None], others[pos:]])
        got = V.adjudicate_block(block, run_c, run_jax, seed=SEED)[pos]
        assert got == alone, \
            f"[{mode}] verdict moved when the vector sat at position {pos}"

    # ...and without the canonical per-sample re-evaluation it WOULD move, in
    # this direction too, so that step is load-bearing rather than decorative.
    block = np.vstack([target[None], others])
    loose = V.adjudicate_block(block, run_c, run_jax, seed=SEED,
                               canonical_single=False)[0]
    assert loose.onset != alone.onset
    if mode == "batch_cleaner":
        assert loose.agreement == V.CLEAN and alone.agreement != V.CLEAN


# ------------------------------- the real thing, against the C oracle


def _oracle_lib_candidates() -> list[str]:
    cands = [os.environ.get("DALEC_ORACLE_LIB"), str(PKG / ".oracle_runtime")]
    nc = shutil.which("nc-config")
    if nc:
        r = subprocess.run([nc, "--libdir"], capture_output=True, text=True)
        if r.returncode == 0 and r.stdout.strip():
            cands.append(r.stdout.strip())
    return [c for c in cands if c and Path(c).is_dir()]


def _oracle_env() -> dict:
    env = dict(os.environ)
    env.pop("JAX_PLATFORMS", None)
    libs = _oracle_lib_candidates()
    if libs:
        env["LD_LIBRARY_PATH"] = os.pathsep.join(
            libs + ([env["LD_LIBRARY_PATH"]] if env.get("LD_LIBRARY_PATH")
                    else []))
    return env


def _unresolved_libraries(env: dict) -> list[str]:
    """Shared objects the loader cannot find, checked BEFORE any oracle run."""
    r = subprocess.run(["ldd", str(ORACLE)], capture_output=True, text=True,
                       env=env)
    if r.returncode != 0:
        return []                       # cannot tell; let the run speak
    return [ln.split("=>")[0].strip() for ln in r.stdout.splitlines()
            if "not found" in ln]


@pytest.fixture(scope="module")
def oracle(tmp_path_factory):
    """Run the C oracle. Skips only if it genuinely cannot be executed.

    A non-zero exit from an oracle that *can* run is a test FAILURE: turning it
    into a skip would let a real oracle regression pass as green, and these are
    the only tests that cover the reported bug end to end.
    """
    if not ORACLE.exists():
        pytest.skip(f"C oracle not built ({ORACLE}) — "
                    "make -C C/projects/JAX_VALIDATION")
    env = _oracle_env()
    missing = _unresolved_libraries(env)
    if missing:
        pytest.skip(f"C oracle cannot load {', '.join(missing)} — set "
                    f"DALEC_ORACLE_LIB or populate .oracle_runtime")
    work = tmp_path_factory.mktemp("oracle")
    state = {"n": 0}

    def run(cbf, params):
        state["n"] += 1
        tag = work / f"r{state['n']}"
        pf = Path(f"{tag}_p.bin")
        params = np.asarray(params, dtype=np.float64)
        params.astype("<f8").tofile(pf)
        pp, ff = Path(f"{tag}_pools.bin"), Path(f"{tag}_flux.bin")
        r = subprocess.run(
            [str(ORACLE), "trajectory", str(cbf), str(pf), str(pp), str(ff)],
            capture_output=True, text=True, env=env)
        assert r.returncode == 0, (
            f"oracle trajectory exited {r.returncode} on {len(params)} "
            f"vectors\nstdout: {r.stdout[-800:]}\nstderr: {r.stderr[-800:]}")
        return (np.fromfile(pp).reshape(len(params), -1, NOPOOLS),
                np.fromfile(ff).reshape(len(params), -1, NOFLUXES))

    return run


@pytest.fixture(scope="module")
def jax_runner():
    import jax
    import netCDF4
    from dalec_jax.model import prederive_vegk, run_dalec_1100
    from dalec_jax.model.dalec_1100 import MET_COLUMNS

    cache: dict[str, object] = {}

    def build(cbf):
        if str(cbf) not in cache:
            with netCDF4.Dataset(str(cbf)) as ds:
                ds.set_auto_mask(False)
                met = {k: np.array(ds[k][:], dtype=float)
                       for k in MET_COLUMNS}
                tix = np.array(ds["time"][:], dtype=float)
                lat = float(ds["LAT"][:])
            deltat = float(tix[1] - tix[0])
            vegk = prederive_vegk(met["DOY"], lat)
            cache[str(cbf)] = jax.jit(jax.vmap(
                lambda p: run_dalec_1100(p, met, lat, deltat, vegk)))
        return cache[str(cbf)]

    def run(cbf, params):
        pools, fluxes = build(cbf)(np.asarray(params, dtype=np.float64))
        return np.asarray(pools), np.asarray(fluxes)

    return run


@pytest.fixture(scope="module")
def companions():
    """The 55 healthy companions, distinct rows of the tracked posterior."""
    import netCDF4
    with netCDF4.Dataset(DATA / "assim_1100.cbr") as ds:
        post = np.array(ds["Parameters"][:], dtype=np.float64)
    return post[list(HEALTHY_ROWS)]


@pytest.mark.parametrize("donor", sorted(DONOR_CBFS))
def test_donor_path_verdict_is_the_same_alone_and_in_a_mixed_batch(
        donor, oracle, jax_runner, companions):
    """The reported bug, at the width it was reported at."""
    cbf = DONOR_CBFS[donor]
    target = np.loadtxt(DATA / "chaos_laplace_modes_1.par.txt")

    def run_c(block):
        return oracle(cbf, block)

    def run_jax(block):
        return jax_runner(cbf, block)

    alone = V.adjudicate_block(target[None], run_c, run_jax, seed=SEED)[0]
    for pos in (0, 27, 55):                       # the original width was 56
        block = np.vstack([companions[:pos], target[None], companions[pos:]])
        assert len(block) == 56
        got = V.adjudicate_block(block, run_c, run_jax, seed=SEED)[pos]
        assert got == alone, (
            f"{donor}: verdict at position {pos} of 56 differs from the "
            f"verdict alone\n  alone: {alone.as_dict()}\n  batch: "
            f"{got.as_dict()}")

    # the divergence is the mortality-factor branch flip on BOTH paths, not a
    # ULP difference on the 1.4e9 soil-energy pool the first report named
    assert alone.agreement in (V.CHAOS_CERTIFIED, V.DISCREPANT)
    assert alone.divergence.name == "nonleaf_mortality_factor"
    assert (alone.divergence.c_value, alone.divergence.jax_value) == (0.0, 1.0)
    assert alone.plausibility.plausible is False, \
        "the stand is dead at the onset; that must be recorded explicitly"
    assert alone.plausibility.live_carbon < 1e-12


def test_c_oracle_is_already_per_sample_deterministic(oracle, companions):
    """Why only the JAX side is canonicalised: oracle_1100 zeroes buffers."""
    cbf = DONOR_CBFS["2004-07"]
    target = np.loadtxt(DATA / "chaos_laplace_modes_1.par.txt")
    block = np.vstack([companions[:2], target[None]])
    bp, bf = oracle(cbf, block)
    sp, sf = oracle(cbf, target[None])
    assert np.array_equal(bp[2].view(np.uint64), sp[0].view(np.uint64))
    assert np.array_equal(bf[2].view(np.uint64), sf[0].view(np.uint64))


def test_jax_is_not_bit_identical_across_batch_widths(jax_runner, companions):
    """The premise of the canonical single-vector rule, measured not assumed."""
    cbf = DONOR_CBFS["2004-07"]
    block = companions[:8]
    bp, _ = jax_runner(cbf, block)
    moved = sum(not np.array_equal(bp[i].view(np.uint64),
                                   jax_runner(cbf, block[i:i + 1])[0][0]
                                   .view(np.uint64))
                for i in range(len(block)))
    assert moved > 0, (
        "no sample moved between batch widths — if XLA has become "
        "width-stable the canonical re-run is no longer load-bearing and this "
        "module's docstring should say so")
