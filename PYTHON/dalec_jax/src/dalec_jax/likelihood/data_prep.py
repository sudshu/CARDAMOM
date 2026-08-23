"""CBF observation loading with exact C semantics.

Replicates READ_NETCDF_TIMESERIES_OBS_FIELDS / READ_NETCDF_SINGLE_OBS_FIELDS
+ TIMESERIES_OBS_STRUCT_PREPROCESS (CARDAMOM_LIKELIHOOD_FUNCTION.c:53-197)
and the relevant slices of CARDAMOM_READ_NETCDF_DATA.c, in numpy at load
time. Everything here is STATIC per CBF — filter gathers become concrete
index arrays; only model output is traced in the jax likelihood.

C sentinel semantics preserved: missing variable -> length 0; missing
attribute -> -9999(.0); default_int/double_value replace -9999 with the
default; min_threshold default is -inf; unc backfill by opt_unc_type; the
structural-error quadrature runs UNCONDITIONALLY on valid indices (so a
stream with no unc info gets sqrt((-9999)^2) = 9999 — harmless for filters
that never read unc[], and reproduced faithfully).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DEFAULT = -9999.0

TIMESERIES_NAMES = ("ABGB", "CH4", "CWOO", "DOM", "ET", "LE", "H", "EWT",
                    "FIR", "GPP", "LAI", "NBE", "ROFF", "SCF", "SIF", "SWE")
SINGLE_NAMES = ("Mean_ABGB", "Mean_GPP", "Mean_LAI", "Mean_FIR",
                "PEQ_NBEmrg", "PEQ_Cefficiency", "PEQ_CUE", "PEQ_C3frac",
                "PEQ_Vcmax25", "PEQ_iniSOM", "PEQ_iniSnow", "PEQ_LCMA",
                "PEQ_clumping", "PEQ_r_ch4", "PEQ_S_fv", "PEQ_rhch4_rhco2")


@dataclass
class TimeseriesObs:
    name: str
    values: np.ndarray            # raw, with -9999 sentinels
    unc: np.ndarray               # backfilled + structural quadrature
    valid_idx: np.ndarray         # int indices of values != -9999
    opt_unc_type: int = 0
    opt_normalization: int = 0
    opt_filter: int = 0
    min_threshold: float = -np.inf
    single_monthly_unc: float = DEFAULT
    single_annual_unc: float = DEFAULT
    single_decadal_unc: float = DEFAULT
    single_mean_unc: float = DEFAULT
    trend_unc: float = DEFAULT
    single_unc: float = DEFAULT
    structural_unc: float = 0.0

    @property
    def valid_obs_length(self) -> int:
        return len(self.valid_idx)


@dataclass
class SingleObs:
    name: str
    value: float = DEFAULT
    unc: float = DEFAULT
    opt_unc_type: int = 0
    min_threshold: float = -np.inf

    @property
    def active(self) -> bool:      # the LIKELIHOOD gate (value != DEFAULT)
        return self.value != DEFAULT


def _attr(var, name, default=DEFAULT):
    try:
        return float(var.getncattr(name))
    except (AttributeError, KeyError):
        return default


def _int_attr(var, name):
    v = _attr(var, name)
    return -9999 if v == DEFAULT else int(v)


def read_timeseries_obs(ds, name: str) -> TimeseriesObs:
    if name in ds.variables:
        var = ds[name]
        values = np.array(var[:], dtype=np.float64)
        uncname = name + "unc"
        if uncname in ds.variables:
            unc = np.array(ds[uncname][:], dtype=np.float64)
        else:
            unc = np.full_like(values, DEFAULT)
        opt_unc_type = _int_attr(var, "opt_unc_type")
        opt_normalization = _int_attr(var, "opt_normalization")
        opt_filter = _int_attr(var, "opt_filter")
        min_threshold = _attr(var, "min_threshold")
        single_monthly_unc = _attr(var, "single_monthly_unc")
        single_annual_unc = _attr(var, "single_annual_unc")
        single_decadal_unc = _attr(var, "single_decadal_unc")
        single_mean_unc = _attr(var, "single_mean_unc")
        trend_unc = _attr(var, "trend_unc")
        single_unc = _attr(var, "single_unc")
        structural_unc = _attr(var, "structural_unc")
    else:
        values = np.zeros(0)
        unc = np.zeros(0)
        opt_unc_type = opt_normalization = opt_filter = -9999
        min_threshold = single_monthly_unc = single_annual_unc = DEFAULT
        single_decadal_unc = single_mean_unc = trend_unc = DEFAULT
        single_unc = structural_unc = DEFAULT

    # TIMESERIES_OBS_STRUCT_PREPROCESS defaults (C lines 58-62)
    if opt_unc_type == -9999:
        opt_unc_type = 0
    if opt_normalization == -9999:
        opt_normalization = 0
    if opt_filter == -9999:
        opt_filter = 0
    if min_threshold == DEFAULT:
        min_threshold = -np.inf
    if structural_unc == DEFAULT:
        structural_unc = 0.0

    valid_idx = np.where(values != DEFAULT)[0].astype(np.int64)

    unc = unc.copy()
    for k in valid_idx:                       # backfill (C lines 87-96)
        if unc[k] == DEFAULT:
            if opt_unc_type < 2:
                unc[k] = single_unc
            elif opt_unc_type == 2:
                a = single_unc * values[k]
                b = single_unc * min_threshold
                unc[k] = a if a > b else b    # C max macro: a<b ? b : a
    for k in valid_idx:                       # quadrature (C lines 102-104)
        unc[k] = np.sqrt(unc[k] ** 2 + structural_unc ** 2)

    return TimeseriesObs(name, values, unc, valid_idx, opt_unc_type,
                         opt_normalization, opt_filter, min_threshold,
                         single_monthly_unc, single_annual_unc,
                         single_decadal_unc, single_mean_unc, trend_unc,
                         single_unc, structural_unc)


def read_single_obs(ds, name: str) -> SingleObs:
    if name in ds.variables:
        var = ds[name]
        value = float(np.array(var[:]))
        if np.isnan(value):
            value = DEFAULT
        unc = _attr(var, "unc")
        opt_unc_type = _int_attr(var, "opt_unc_type")
        if opt_unc_type == -9999:
            opt_unc_type = 0
        min_threshold = _attr(var, "min_threshold")
        if min_threshold == DEFAULT:
            min_threshold = -np.inf
    else:
        value, unc, opt_unc_type, min_threshold = DEFAULT, DEFAULT, 0, -np.inf
    return SingleObs(name, value, unc, opt_unc_type, min_threshold)


@dataclass
class CbfData:
    met: dict
    time: np.ndarray
    LAT: float
    deltat: float
    EDC: int
    ts: dict = field(default_factory=dict)      # name -> TimeseriesObs
    single: dict = field(default_factory=dict)  # name -> SingleObs
    skt_ref_mean: float = 0.0
    edc_eqf: float = 2.0

    @property
    def n_timesteps(self) -> int:
        return len(self.time)


MET_NAMES = ("SSRD", "T2M_MIN", "T2M_MAX", "CO2", "DOY", "TOTAL_PREC", "VPD",
             "BURNED_AREA", "SNOWFALL", "SKT", "STRD", "DISTURBANCE_FLUX",
             "YIELD")


def load_cbf(path) -> CbfData:
    import netCDF4

    with netCDF4.Dataset(path) as ds:
        ds.set_auto_mask(False)                 # keep -9999 sentinels raw
        met = {k: np.array(ds[k][:], dtype=np.float64) for k in MET_NAMES}
        time = np.array(ds["time"][:], dtype=np.float64)
        LAT = float(np.array(ds["LAT"][:]))
        EDC = int(float(np.array(ds["EDC"][:]))) if "EDC" in ds.variables else 1
        edc_eqf = (float(np.array(ds["EDC_EQF"][:]))
                   if "EDC_EQF" in ds.variables else 2.0)

        skt_attr = None
        if "reference_mean" in ds["SKT"].ncattrs():
            skt_attr = float(ds["SKT"].getncattr("reference_mean"))

        out = CbfData(met=met, time=time, LAT=LAT,
                      deltat=float(time[1] - time[0]), EDC=EDC,
                      edc_eqf=edc_eqf)
        for nm in TIMESERIES_NAMES:
            out.ts[nm] = read_timeseries_obs(ds, nm)
        for nm in SINGLE_NAMES:
            out.single[nm] = read_single_obs(ds, nm)

    if skt_attr is not None:
        out.skt_ref_mean = skt_attr
    else:                                       # C sequential per-term mean
        acc = 0.0
        n = len(met["SKT"])
        for v in met["SKT"]:
            acc += float(v) / n
        out.skt_ref_mean = acc
    return out
