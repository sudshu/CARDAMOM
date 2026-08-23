"""The 15 live DALEC_1100 EDCs + the RUN_DALEC_EDCs recording semantics.

Payload constants transcribed from DALEC_1100_MODCONFIG (DALEC_1100.c:
1183-1427); implementations from C/projects/.../DALEC_EDCs/*.c.

Recording semantics (RUN_DALEC_EDCs, DALEC_EDC_FUNCTIONS.c:24-60): within a
phase (prerun / postrun), EDCs are visited in INDEX order; each M_EDCs slot
is written only while the running phase sum P is > -inf (NaN also stops —
NaN > -inf is False). Slots after the first -inf/NaN stay 0 (per-sample-
zeroed buffers). Under jit every EDC value is computed; the recording mask
reproduces what the C writes.
"""
from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from ..constants import DGCM_TK0C
from ..indices import E, F, NOEDCS, NOPOOLS, P, PARMAX, S
from . import sio

# ---------------- payload constants (MODCONFIG) ----------------

_INEQ = {  # E-index -> (big parameter, small parameter)
    E.litcwdtor: (P.t_lit, P.t_cwd),
    E.cwdsomtor: (P.t_cwd, P.t_som),
    E.rootwoodtor: (P.t_root, P.t_wood),
    E.mr_rates: (P.rauto_mr_r, P.rauto_mr_w),
    E.fol2lig_cf: (P.cf_foliar, P.cf_ligneous),
    E.relativepsi50: (P.psi_50HMF, P.psi_50),
}
_VCMAX_LCMA = (P.Vcmax25, P.LCMA, 1.0399, 0.1956)   # TRY database prior

TRAJ_POOLS = [S.C_lab, S.C_fol, S.C_roo, S.C_woo, S.C_cwd, S.C_lit, S.C_som,
              S.H2O_LY1, S.H2O_LY2, S.H2O_LY3, S.H2O_SWE,
              S.E_LY1, S.E_LY2, S.E_LY3]
_ETOL = 0.1

PRERUN_MASK = tuple(i in set(_INEQ) | {E.vcmax_lcma} for i in range(NOEDCS))


def state_range_bounds(pars) -> tuple[jnp.ndarray, jnp.ndarray]:
    """min/max per pool (DALEC_1100.c:1246-1326). NOTE the C quirks kept:
    D_LF_LY2 is left unbounded (only LY1 and LY3 get [0,1]); H2O/E pools get
    min 0 with no max; D_PSI_* and M_LAI_* are unbounded."""
    lo = np.full(NOPOOLS, -np.inf)
    hi = np.full(NOPOOLS, np.inf)
    lo[[S.C_lab, S.C_fol, S.C_roo, S.C_woo, S.C_cwd, S.C_lit, S.C_som]] = 0
    hi[S.C_lab] = PARMAX[P.i_labile]
    hi[S.C_fol] = PARMAX[P.i_foliar]
    hi[S.C_roo] = PARMAX[P.i_root]
    hi[S.C_woo] = PARMAX[P.i_wood]
    hi[S.C_cwd] = PARMAX[P.i_cwd]
    hi[S.C_lit] = PARMAX[P.i_lit]
    hi[S.C_som] = PARMAX[P.i_som]
    lo[[S.H2O_LY1, S.H2O_LY2, S.H2O_LY3, S.H2O_SWE]] = 0
    lo[[S.E_LY1, S.E_LY2, S.E_LY3]] = 0
    lo[S.D_LAI] = 0
    hi[S.D_LAI] = PARMAX[P.lambda_max]
    lo[S.D_SCF] = 0
    hi[S.D_SCF] = 1
    lo[[S.D_TEMP_LY1, S.D_TEMP_LY2, S.D_TEMP_LY3]] = 173.15
    hi[[S.D_TEMP_LY1, S.D_TEMP_LY2, S.D_TEMP_LY3]] = 373.15
    lo[S.D_LF_LY1] = 0
    hi[S.D_LF_LY1] = 1
    lo[S.D_LF_LY3] = 0            # LY2 deliberately absent in the C
    hi[S.D_LF_LY3] = 1
    lo[[S.D_SM_LY1, S.D_SM_LY2, S.D_SM_LY3]] = 0
    hi[[S.D_SM_LY1, S.D_SM_LY2, S.D_SM_LY3]] = 1
    return jnp.asarray(lo), jnp.asarray(hi)


# ---------------- helpers (mean_pool.c / mean_flux.c: per-term division) ---

def _mean_over_time(arr2d, index, nc):
    # C: meanpool += A[c*width+index]/nc  — per-term division preserved
    return jnp.sum(arr2d[:nc, index] / nc)


# ---------------- individual EDC values ----------------

def edc_values(pars, pools, fluxes, cfg) -> jnp.ndarray:
    """All 15 EDC values, unconditionally evaluated. pools (T+1,30),
    fluxes (T,100). cfg: dict(edc_eqf, skt_ref_mean, dint, n_timesteps)."""
    T = cfg["n_timesteps"]
    vals = [None] * NOEDCS

    # prerun: parameter inequalities + log-ratio prior
    for ei, (big, small) in _INEQ.items():
        vals[ei] = jnp.where(pars[big] < pars[small], -jnp.inf, 0.0)
    num, den, mu, sd = _VCMAX_LCMA
    mod_ratio = jnp.log(pars[num]) / jnp.log(pars[den])
    r = (mod_ratio - mu) / sd
    vals[E.vcmax_lcma] = -0.5 * r * r

    # state_ranges: rows 0..T-1 ONLY (the C never checks the final row),
    # violation on strict compare; NaN passes (both compares False), exactly
    # as in the C while-loop.
    lo, hi = state_range_bounds(pars)
    seg = pools[:T]
    viol = jnp.logical_or(seg < lo[None, :], seg > hi[None, :]).any()
    vals[E.state_ranges] = jnp.where(viol, -jnp.inf, 0.0)

    # state_trajectories (DALEC_EDC_TRAJECTORY.c)
    FT = jnp.sum(fluxes[:T], axis=0)          # per-flux total over time
    dint = cfg["dint"]                        # C int, precomputed outside
    nterm = T // dint + 1                     # C integer division
    eqf = cfg["edc_eqf"]
    pedc = jnp.asarray(0.0)
    for pidx in TRAJ_POOLS:
        mjan = jnp.sum(pools[(np.arange(nterm) * dint), pidx] / nterm)
        fin = jnp.asarray(0.0)
        fout = jnp.asarray(0.0)
        for fi in sio.state_input_fluxes(pidx):
            ft = FT[fi]
            fout = fout + jnp.where(ft < 0, ft, 0.0)
            fin = fin + jnp.where(ft < 0, 0.0, ft)
        for fi in sio.state_output_fluxes(pidx):
            ft = FT[fi]
            fin = fin + jnp.where(ft < 0, ft, 0.0)
            fout = fout + jnp.where(ft < 0, 0.0, ft)
        pstart = pools[0, pidx]
        rm = fin / fout
        rs = rm * mjan / pstart
        pedc = pedc + (-0.5 * jnp.power(jnp.log(rs) / jnp.log(eqf), 2)
                       - 0.5 * jnp.power(jnp.log(rs / rm)
                                         / jnp.log(1 + _ETOL), 2))
    vals[E.state_trajectories] = pedc

    # nsc_ratio (DALEC_EDC_NSC_ABGB_RATIO.c)
    m_nsc = _mean_over_time(pools, S.C_lab, T + 1)
    m_else = (_mean_over_time(pools, S.C_fol, T + 1)
              + _mean_over_time(pools, S.C_roo, T + 1)
              + _mean_over_time(pools, S.C_woo, T + 1))
    r = (m_nsc / (m_nsc + m_else) - 0.1) / 0.05
    vals[E.nsc_ratio] = -0.5 * r * r

    # cfcr_ratio (POOL_RATIO): log(Mfol/Mroo)/log(2)
    r = jnp.log(_mean_over_time(pools, S.C_fol, T + 1)
                / _mean_over_time(pools, S.C_roo, T + 1)) / jnp.log(2.0)
    vals[E.cfcr_ratio] = -0.5 * r * r

    # fffr_ratio (FLUX_RATIO): log(mean foliar_prod / mean root_prod)/log(2)
    r = jnp.log(_mean_over_time(fluxes, F.foliar_prod, T)
                / _mean_over_time(fluxes, F.root_prod, T)) / jnp.log(2.0)
    vals[E.fffr_ratio] = -0.5 * r * r

    # mean layer temperatures vs SKT reference mean, sigma 5 K
    for ei, sidx in ((E.mean_ly1_temp, S.D_TEMP_LY1),
                     (E.mean_ly2_temp, S.D_TEMP_LY2),
                     (E.mean_ly3_temp, S.D_TEMP_LY3)):
        mt = _mean_over_time(pools, sidx, T + 1)
        r = (mt - cfg["skt_ref_mean"] - DGCM_TK0C) / 5
        vals[ei] = -0.5 * r * r

    assert all(v is not None for v in vals)
    return jnp.stack([jnp.asarray(v, dtype=jnp.float64) for v in vals])


def record_phase(values, prerun: bool, start_P):
    """RUN_DALEC_EDCs recording: index order, write while P > -inf.
    Returns (recorded (15,), phase P sum)."""
    recorded = []
    Pacc = start_P
    for i in range(NOEDCS):
        if PRERUN_MASK[i] != prerun:
            recorded.append(jnp.asarray(0.0))
            continue
        write = Pacc > -jnp.inf          # False for -inf AND NaN, as in C
        recorded.append(jnp.where(write, values[i], 0.0))
        Pacc = jnp.where(write, Pacc + values[i], Pacc)
    return jnp.stack(recorded), Pacc


def compute_dint(time_index: np.ndarray) -> int:
    """C: (int)floor(N/(TIME[N-1]-TIME[0])*365.25) — plain python floats."""
    n = len(time_index)
    return int(np.floor(n / (float(time_index[-1]) - float(time_index[0]))
                        * 365.25))


def skt_reference_mean(skt: np.ndarray, attr_value: float | None) -> float:
    """CBF attribute if present, else the C's sequential per-term mean."""
    if attr_value is not None:
        return float(attr_value)
    acc = 0.0
    n = len(skt)
    for v in skt:                        # sequential += v/n, like the C
        acc += float(v) / n
    return acc
