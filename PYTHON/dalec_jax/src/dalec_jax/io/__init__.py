"""Common xarray output schema — the analysis layer consumes this from
EITHER engine (C oracle dumps or JAX arrays), which is what makes
paper-analysis equivalence a like-for-like comparison."""
from __future__ import annotations

import numpy as np

from ..indices import FLUX_NAMES, POOL_NAMES


def to_xarray(pools: np.ndarray, fluxes: np.ndarray, pars: np.ndarray,
              time_days: np.ndarray, engine: str):
    """pools (S, T+1, 30), fluxes (S, T, 100), pars (S, 89) -> xr.Dataset."""
    import xarray as xr

    S, Tp1, _ = pools.shape
    T = Tp1 - 1
    return xr.Dataset(
        {
            "POOLS": (("sample", "time_pools", "pool"), pools),
            "FLUXES": (("sample", "time", "flux"), fluxes),
            "PARS": (("sample", "parameter"), pars),
        },
        coords={
            "pool": list(POOL_NAMES),
            "flux": list(FLUX_NAMES),
            "time": np.asarray(time_days, dtype=float),
            "sample": np.arange(S),
        },
        attrs={"engine": engine, "model": "DALEC_1100"},
    )
