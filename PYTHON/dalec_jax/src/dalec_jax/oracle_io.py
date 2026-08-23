"""Load golden fixtures produced by tools/gen_fixtures.py + comparison helpers."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parent.parent.parent
GOLDEN = PKG / "tests/_golden"


def manifest() -> dict:
    return json.loads((GOLDEN / "manifest.json").read_text())


def module_spec(name: str) -> dict:
    for spec in manifest()["oracle_manifest"]["modules"]:
        if spec["name"] == name:
            return spec
    raise KeyError(name)


def module_golden(name: str) -> tuple[np.ndarray, np.ndarray]:
    """(inputs (n, n_in), outputs (n, n_out)) for one module."""
    spec = module_spec(name)
    x = np.fromfile(GOLDEN / f"modules/{name}.in.bin").reshape(
        -1, len(spec["inputs"]))
    y = np.fromfile(GOLDEN / f"modules/{name}.out.bin").reshape(
        -1, len(spec["outputs"]))
    return x, y


def trajectory_golden() -> dict:
    tdir = GOLDEN / "trajectories"
    meta = json.loads((tdir / "fixture_meta.json").read_text())
    params = np.fromfile(tdir / "fixture_params.bin").reshape(len(meta["rows"]), -1)
    n = params.shape[0]
    pools = np.fromfile(tdir / "pools.bin").reshape(n, -1, 30)
    fluxes = np.fromfile(tdir / "fluxes.bin").reshape(n, -1, 100)
    mlf = np.fromfile(tdir / "mlf.bin").reshape(n, -1)
    return {"meta": meta, "params": params, "pools": pools,
            "fluxes": fluxes, "mlf": mlf}


def ulp_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """ULP distance for float64; identical non-finites count as 0,
    mismatched non-finites as 2**62. +0/-0 differ by 1."""
    a = np.ascontiguousarray(a, dtype=np.float64)
    b = np.ascontiguousarray(b, dtype=np.float64)
    ia = a.view(np.int64).copy()
    ib = b.view(np.int64).copy()
    ia = np.where(ia < 0, np.int64(-(2 ** 63)) - ia - 1, ia)
    ib = np.where(ib < 0, np.int64(-(2 ** 63)) - ib - 1, ib)
    d = np.abs(ia - ib)
    both_fin = np.isfinite(a) & np.isfinite(b)
    same_nonfin = (~np.isfinite(a) & ~np.isfinite(b)
                   & (np.isnan(a) == np.isnan(b))
                   & (np.isnan(a) | (np.sign(a) == np.sign(b))))
    return np.where(both_fin, d, np.where(same_nonfin, 0, np.int64(2 ** 62)))


def rel_err(ref: np.ndarray, got: np.ndarray) -> np.ndarray:
    """|got-ref| / max(|ref|, tiny); identical non-finites -> 0, else inf."""
    both_fin = np.isfinite(ref) & np.isfinite(got)
    same_nonfin = (~np.isfinite(ref) & ~np.isfinite(got)
                   & (np.isnan(ref) == np.isnan(got))
                   & (np.isnan(ref) | (np.sign(ref) == np.sign(got))))
    denom = np.maximum(np.abs(ref), 1e-300)
    e = np.abs(got - ref) / denom
    return np.where(both_fin, e, np.where(same_nonfin, 0.0, np.inf))
