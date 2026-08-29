"""C/JAX trajectory adjudication: batch-invariant, magnitude- and ULP-aware.

Reusable library infrastructure. It is the one tracked copy of the L4 verdict,
shared by `tests/test_trajectory.py` and `tools/gen_fixtures.py` (which use the
element criterion) and intended for the run-level
`runs/*/verify_against_c.py`. **No tracked runtime path calls
:func:`adjudicate_block` yet**: the run verifier is untracked, still builds its
dithers from one batch-indexed stream, and therefore still carries the defect
this module fixes. Landing the library does not fix the pipeline; the migration
is tracked in plan.md P9.

It runs neither the model nor the oracle — callers pass ``run_c`` / ``run_jax``
callables — so the logic is testable without either.

The element criterion is the repository's locked L4 contract and is unchanged
(tests/TOLERANCES.md "Trajectory chaos"):

    |jax - c| <= 1e-10 * max(|c|, RMS_var)   OR   |jax - c| <= 1e-12

and the certification rule is the locked one, also unchanged: a sample
certifies when the JAX divergence onset is no earlier than the C's own
self-divergence onset minus ``CHAOS_MARGIN`` steps. This module adds no second
route to a certificate. What it changes is how the inputs to that rule are
obtained. Measurements and defect history: CHANGELOG.md (2026-08-29).

**Two orthogonal axes.** A verdict carries a numerical *agreement* statement
and, separately, a physical *plausibility* statement. Plausibility never
softens agreement: the caller decides, with the reason recorded, whether a
vector whose state has left the physically meaningful range should be
interpreted at all. Folding plausibility into a tolerance is how a real
transcription bug gets hidden.

**Batch invariance.** The verdict is a function of the parameter vector and the
forcing path, nothing else. :func:`dither_block` seeds from the vector's own
bits, so the perturbations no longer depend on its ordinal position in
whichever block it was evaluated with. :func:`adjudicate_block` evaluates
*every* sample alone, because ``jit(vmap(...))`` is not bit-identical across
batch widths in either direction — a sample can look clean in a block and dirty
alone as readily as the reverse. The C oracle needs no such treatment, as
``zero_model_buffers`` makes it per-sample deterministic.

**Dither depth.** :func:`adjudicate` may escalate ``k`` -> ``k_escalated``
before calling a sample a genuine discrepancy. Escalation is *monotone and
one-sided*: ``c_self`` is a minimum over K onsets, so raising K can only move
DISCREPANT -> CHAOS_CERTIFIED, never the reverse, and nothing here controls the
false-positive rate of that move. It is a deeper measurement of the same rule,
not a weaker rule, and it is not what repairs either motivating path.

**What certification does NOT cover.** The verdict is keyed on the *first*
divergence onset. Once a sample certifies, nothing downstream of that step is
examined, so a distinct, unrelated error introduced at a later step is
invisible to it — planted late-step mutations produce a byte-identical
:class:`Verdict`. This is a property of the locked rule (tests/TOLERANCES.md
"Trajectory chaos"), not of this implementation, and it is why a certificate is
evidence about one onset rather than a whole-trajectory guarantee.

**Reporting.** :class:`DivergentElement` names the failing variable and carries
|delta|, the scale the criterion divided by, the relative excess and the ULP
distance — enough to tell last-bit noise from a branch flip at a glance.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import asdict, dataclass, field

import numpy as np

from .indices import FLUX_NAMES, POOL_NAMES, S
from .oracle_io import ulp_distance

__all__ = [
    "REL_TOL", "ABS_TOL", "CHAOS_MARGIN", "DITHER_K", "DITHER_K_ESCALATED",
    "LIVE_CARBON_FLOOR", "CLEAN", "CHAOS_CERTIFIED", "DISCREPANT",
    "DivergentElement", "Plausibility", "Verdict",
    "element_fail", "divergence_step", "break_step", "divergent_elements",
    "first_divergence", "dither_block", "state_plausibility",
    "adjudicate", "adjudicate_block",
]

#: L4 element criterion (tests/TOLERANCES.md). Not negotiable here.
REL_TOL = 1e-10
ABS_TOL = 1e-12

#: A JAX onset this many steps before the C's own self-divergence onset is
#: still consistent with 1-ULP sensitivity.
CHAOS_MARGIN = 10

#: All-parameter 1-ULP dithers per vector, and the single escalation.
DITHER_K = 8
DITHER_K_ESCALATED = 64

#: Live carbon (C_lab + C_fol + C_roo + C_woo, gC m-2) below which the stand is
#: numerically dead: every guard that tests a live pool against zero is then
#: decided by rounding, not by the model. One microgram of carbon per square
#: metre sits ~5.7 orders below the last physically meaningful value in a
#: collapsing DALEC trajectory (0.51 gC m-2) and ~11 orders above what the same
#: trajectory reaches three steps later (6.7e-17 gC m-2).
LIVE_CARBON_FLOOR = 1e-6

_LIVE_CARBON_POOLS = (S.C_lab, S.C_fol, S.C_roo, S.C_woo)


# ----------------------------------------------------------- element gate


def element_fail(c: np.ndarray, j: np.ndarray) -> np.ndarray:
    """The locked L4 element criterion, per element of a (T, nvar) block.

    ``|jax - c| <= REL_TOL * max(|c|, RMS_var)`` OR ``|jax - c| <= ABS_TOL``.

    Non-finite handling is inherited verbatim from the locked criterion and is
    coarser than it looks: a pair passes whenever BOTH sides are non-finite,
    whichever non-finite values those are, so ``(+inf, nan)`` and
    ``(+inf, -inf)`` both pass. Only a finite/non-finite mismatch fails here.
    Exactness of the -inf sentinels is enforced separately, by the EDC gate.
    """
    c = np.asarray(c, dtype=np.float64)
    j = np.asarray(j, dtype=np.float64)
    fin = np.isfinite(c)
    with np.errstate(all="ignore"):
        rms = np.sqrt(np.nanmean(np.where(fin, c, np.nan) ** 2, axis=0))
        rms = np.nan_to_num(rms, nan=1.0)
        mixed = np.abs(j - c) / np.maximum(np.abs(np.where(fin, c, 0.0)),
                                           np.maximum(rms, 1e-300))
    ok = (mixed <= REL_TOL) | (np.abs(j - c) <= ABS_TOL)
    return np.where(fin & np.isfinite(j), ~ok,
                    ~((~np.isfinite(c)) & (~np.isfinite(j))))


def _scale(c: np.ndarray) -> np.ndarray:
    """The per-ELEMENT denominator the criterion uses, ``max(|c|, RMS_var)``.

    Same shape as ``c``: the column RMS is a floor under each element's own
    magnitude, which is what makes the gate scale-relative at both ends of the
    dynamic range.
    """
    c = np.asarray(c, dtype=np.float64)
    fin = np.isfinite(c)
    with np.errstate(all="ignore"):
        rms = np.sqrt(np.nanmean(np.where(fin, c, np.nan) ** 2, axis=0))
    rms = np.nan_to_num(rms, nan=1.0)
    return np.maximum(np.abs(np.where(fin, c, 0.0)),
                      np.maximum(rms, 1e-300))


def divergence_step(cp, jp, cf, jf) -> int:
    """First step at which any pool or flux element fails; -1 if none.

    The C writes one more pool row than flux row per sample (initial state plus
    T steps) and the flux mask is padded to match; that shape relation is
    checked rather than assumed.
    """
    cp = np.asarray(cp)
    cf = np.asarray(cf)
    if cf.shape[0] != cp.shape[0] - 1:
        raise ValueError(
            f"expected one fewer flux row than pool rows, got {cf.shape[0]} "
            f"flux rows and {cp.shape[0]} pool rows")
    rows = element_fail(cp, jp).any(axis=1) | np.concatenate(
        [element_fail(cf, jf).any(axis=1), [False]])
    return int(np.argmax(rows)) if rows.any() else -1


def break_step(pools: np.ndarray) -> int:
    """C isfinite-break semantics: first all-zero (calloc) pool row, else -1."""
    zero = np.all(np.asarray(pools) == 0, axis=-1)
    return int(np.argmax(zero)) if zero.any() else -1


# ------------------------------------------------ magnitude / ULP report


@dataclass(frozen=True)
class DivergentElement:
    """One element that failed the L4 criterion, with its magnitude context."""

    step: int
    kind: str            # "pool" | "flux"
    index: int
    name: str
    c_value: float
    jax_value: float
    abs_diff: float
    scale: float         # max(|c|, RMS_var): what the criterion divided by
    rel_to_scale: float  # abs_diff / scale, i.e. the criterion's own ratio
    ulps: int


def _element(kind, names, c_block, j_block, step, var) -> DivergentElement:
    c = float(np.asarray(c_block)[step, var])
    j = float(np.asarray(j_block)[step, var])
    sc = float(_scale(c_block)[step, var])
    with np.errstate(all="ignore"):
        rel = float(abs(j - c) / sc) if sc > 0 else float("inf")
    return DivergentElement(
        step=int(step), kind=kind, index=int(var), name=names[var],
        c_value=c, jax_value=j, abs_diff=float(abs(j - c)), scale=sc,
        rel_to_scale=rel,
        ulps=int(np.asarray(ulp_distance(np.float64(c),
                                         np.float64(j))).reshape(-1)[0]))


def divergent_elements(cp, jp, cf, jf, step: int) -> list[DivergentElement]:
    """Every element failing the criterion at ``step``, pools then fluxes."""
    out: list[DivergentElement] = []
    fp = element_fail(cp, jp)
    if 0 <= step < fp.shape[0]:
        for v in np.where(fp[step])[0]:
            out.append(_element("pool", POOL_NAMES, cp, jp, step, v))
    ff = element_fail(cf, jf)
    if 0 <= step < ff.shape[0]:
        for v in np.where(ff[step])[0]:
            out.append(_element("flux", FLUX_NAMES, cf, jf, step, v))
    return out


def _severity(e: DivergentElement):
    """Total order for "worst element", with NaN ranked last.

    A NaN ratio (a non-finite C value against a finite JAX one) must never
    outrank a real full-scale failure. Left to ``max``'s NaN-propagating
    comparisons the answer depends on iteration order, which is precisely the
    "report names the wrong element" failure this module exists to prevent.
    """
    r = e.rel_to_scale
    return (-math.inf if math.isnan(r) else r, e.kind, -e.index)


def first_divergence(cp, jp, cf, jf) -> DivergentElement | None:
    """The worst element at the divergence onset, or None if the run is clean.

    "Worst" is the largest ratio the criterion itself computed, so the report
    names the element that actually failed rather than whichever variable
    happens to carry the largest raw difference — a pool at 1e9 drifting by a
    couple of ULP loses to a bounded indicator flipping 0 -> 1, which is the
    right way round.
    """
    step = divergence_step(cp, jp, cf, jf)
    if step < 0:
        return None
    els = divergent_elements(cp, jp, cf, jf, step)
    if not els:
        return None
    return max(els, key=_severity)


# -------------------------------------------------- per-vector dithering


def _vector_seed(row: np.ndarray, seed: int) -> np.random.SeedSequence:
    """A SeedSequence determined by the vector's raw bits and ``seed`` only."""
    bits = np.ascontiguousarray(row, dtype="<f8").tobytes()
    digest = hashlib.blake2b(bits, digest_size=8,
                             person=b"dalecdit").digest()
    return np.random.SeedSequence([int(seed),
                                   int.from_bytes(digest, "little")])


def dither_block(row: np.ndarray, k: int = DITHER_K,
                 seed: int = 0) -> np.ndarray:
    """``k`` all-parameter 1-ULP dithers of one parameter vector.

    A pure function of ``(row bits, k, seed)`` — never of the vector's position
    in a batch. Rows are drawn in order, so a larger ``k`` extends the block
    rather than replacing it: ``dither_block(row, 64)[:8]`` is
    ``dither_block(row, 8)``. That is what makes escalation monotone.
    """
    row = np.asarray(row, dtype=np.float64)
    rng = np.random.default_rng(_vector_seed(row, seed))
    up = rng.random((int(k), row.shape[0])) < 0.5
    base = np.repeat(row[None, :], int(k), axis=0)
    return np.where(up, np.nextafter(base, np.inf),
                    np.nextafter(base, -np.inf))


# ----------------------------------------------------- plausibility axis


@dataclass(frozen=True)
class Plausibility:
    """A physical statement about the state, kept apart from the verdict."""

    plausible: bool
    reason: str | None
    live_carbon: float
    floor: float


def state_plausibility(pools_c, step: int,
                       live_carbon_floor: float = LIVE_CARBON_FLOOR
                       ) -> Plausibility:
    """Classify the C state at ``step``: physical, or numerically dead.

    This is *not* a tolerance and never changes a verdict. It exists so that a
    vector whose live carbon has collapsed to where a guard testing a pool
    against zero is decided by rounding is rejected for the reason that is
    actually true, instead of being recorded as a C/JAX implementation
    discrepancy.
    """
    pools_c = np.asarray(pools_c, dtype=np.float64)
    if pools_c.ndim != 2 or pools_c.shape[0] == 0:
        return Plausibility(False, "no pool rows to classify", float("nan"),
                            live_carbon_floor)
    if step < 0 or step >= pools_c.shape[0]:
        step = pools_c.shape[0] - 1
    live = float(np.sum(pools_c[step, list(_LIVE_CARBON_POOLS)]))
    if not np.isfinite(live) or abs(live) < live_carbon_floor:
        return Plausibility(
            False,
            (f"live carbon {live:.3e} gC m-2 is below the physical floor "
             f"{live_carbon_floor:.0e} at step {step}: the stand is "
             f"numerically dead, so guards testing a live pool against zero "
             f"are decided by rounding rather than by the model"),
            live, live_carbon_floor)
    return Plausibility(True, None, live, live_carbon_floor)


# ------------------------------------------------------------- adjudication

CLEAN = "clean"
CHAOS_CERTIFIED = "chaos_certified"
DISCREPANT = "discrepant"


def _json_safe(value):
    """Strictly JSON-representable: non-finite floats become ``None``."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


@dataclass(frozen=True)
class Verdict:
    """Numerical agreement plus, separately, physical plausibility."""

    agreement: str                      # clean | chaos_certified | discrepant
    onset: int
    break_step_c: int
    break_step_jax: int
    c_self_onset: int = -1
    n_dithers: int = 0
    n_dithers_diverged: int = 0
    divergence: DivergentElement | None = None
    plausibility: Plausibility | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def blocks_interpretation(self) -> bool:
        """Fail-closed: a discrepancy the C's own 1-ULP onset cannot explain."""
        return self.agreement == DISCREPANT

    def as_dict(self) -> dict:
        """A report dict that ``json.dumps(..., allow_nan=False)`` accepts.

        Non-finite floats (an infinite ``rel_to_scale`` at zero scale, a NaN
        ``live_carbon``) become ``None``; the dataclass keeps the raw values.
        """
        return _json_safe(asdict(self))


def adjudicate(params, pools_c, fluxes_c, pools_jax, fluxes_jax, run_c, *,
               seed: int = 0, k: int = DITHER_K,
               k_escalated: int = DITHER_K_ESCALATED,
               margin: int = CHAOS_MARGIN,
               live_carbon_floor: float = LIVE_CARBON_FLOOR) -> Verdict:
    """Adjudicate ONE sample. ``run_c(block) -> (pools, fluxes)``.

    ``run_c`` is only called if the sample is not clean, and then at most twice
    (``k`` dithers, then ``k_escalated`` if the small block did not certify).
    Everything it is asked to run is derived from ``params`` alone, so the
    verdict does not depend on any batch this sample came from.
    """
    params = np.asarray(params, dtype=np.float64).reshape(-1)
    bc, bj = break_step(pools_c), break_step(pools_jax)
    onset = divergence_step(pools_c, pools_jax, fluxes_c, fluxes_jax)

    if onset < 0:
        if bc == bj:
            return Verdict(CLEAN, onset, bc, bj)
        return Verdict(DISCREPANT, onset, bc, bj,
                       notes=(f"pointwise clean but C breaks at {bc} and JAX "
                              f"at {bj}",))

    div = first_divergence(pools_c, pools_jax, fluxes_c, fluxes_jax)
    plaus = state_plausibility(pools_c, onset, live_carbon_floor)
    notes: list[str] = []
    if bc != bj:
        notes.append(f"break step differs: C {bc} vs JAX {bj}")

    verdict = None
    for n_dith in ([k] if k >= k_escalated else [k, k_escalated]):
        dp, df = run_c(dither_block(params, n_dith, seed))
        dp = np.asarray(dp, dtype=np.float64)
        df = np.asarray(df, dtype=np.float64)
        # per-dither step at which the dithered C leaves the base C
        onsets = np.array([divergence_step(pools_c, dp[m], fluxes_c, df[m])
                           for m in range(n_dith)], dtype=int)
        pos = onsets[onsets >= 0]
        c_self = int(pos.min()) if pos.size else -1
        certified = c_self >= 0 and onset >= c_self - margin

        verdict = Verdict(
            CHAOS_CERTIFIED if certified else DISCREPANT, onset, bc, bj,
            c_self_onset=c_self, n_dithers=n_dith,
            n_dithers_diverged=int(pos.size), divergence=div,
            plausibility=plaus, notes=tuple(notes))
        if certified:
            break
        if n_dith != k_escalated:
            notes.append(
                f"K={n_dith} dithers did not certify; escalated to "
                f"K={k_escalated} before reporting a genuine discrepancy")
    return verdict


def adjudicate_block(params, run_c, run_jax, *, seed: int = 0,
                     k: int = DITHER_K, k_escalated: int = DITHER_K_ESCALATED,
                     margin: int = CHAOS_MARGIN,
                     live_carbon_floor: float = LIVE_CARBON_FLOOR,
                     canonical_single: bool = True) -> list[Verdict]:
    """Adjudicate a block of vectors, batch-invariantly.

    ``run_jax`` is vmapped by its caller and XLA does not promise bit-identical
    codegen across batch widths, so EVERY sample is re-run alone and adjudicated
    on that canonical trajectory — including samples that looked clean in a
    block pass. Canonicalising only the dirty ones leaves the guarantee
    one-sided: a sample that is clean at width 56 and divergent at width 1 would
    be recorded CLEAN, which is the same position-dependence this module exists
    to remove, mirrored. Set ``canonical_single=False`` only to measure the
    block-vs-single difference.
    """
    params = np.asarray(params, dtype=np.float64)
    cp, cf = (np.asarray(x, dtype=np.float64) for x in run_c(params))
    if not canonical_single:
        jp, jf = (np.asarray(x, dtype=np.float64) for x in run_jax(params))

    def _run_c(block):
        return tuple(np.asarray(x, dtype=np.float64) for x in run_c(block))

    out: list[Verdict] = []
    for i in range(params.shape[0]):
        if canonical_single:
            sp, sf = run_jax(params[i:i + 1])
            p_i = np.asarray(sp, dtype=np.float64)[0]
            f_i = np.asarray(sf, dtype=np.float64)[0]
        else:
            p_i, f_i = jp[i], jf[i]
        out.append(adjudicate(
            params[i], cp[i], cf[i], p_i, f_i, _run_c,
            seed=seed, k=k, k_escalated=k_escalated, margin=margin,
            live_carbon_floor=live_carbon_floor))
    return out
