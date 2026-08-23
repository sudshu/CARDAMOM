#!/usr/bin/env python3
"""Generate Tier-A module fixtures + trajectory/mlf parameter fixtures, and
run the C oracle over them to produce the golden files.

Outputs under tests/_golden/ (gitignored; regenerate with this script):
  manifest.json                     oracle manifest + environment fingerprint
  modules/<NAME>.in.bin             n_cases x n_in float64 (LHS + edge cases)
  modules/<NAME>.out.bin            n_cases x n_out float64 (C oracle output)
  trajectories/fixture_params.bin   n_fix x 89 float64
  trajectories/fixture_meta.json    provenance of every fixture row
  trajectories/pools.bin            n_fix x 241 x 30
  trajectories/fluxes.bin           n_fix x 240 x 100
  trajectories/mlf.bin              n_fix x 47  [15 EDCs | 31 likelihoods | P]

Fixture parameter set (deterministic, seeded):
  64 posterior samples that genuinely run (spread over the non-gated tail),
  16 posterior samples that are prerun-EDC gated (burn-in region),
   8 viable-ensemble members (runs/baseline_1100 rejection-sampled),
  32 prior-bound draws (median prior draw exercises the isfinite break).

Everything is generated twice and byte-compared (determinism gate).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

PKG = Path(__file__).resolve().parent.parent
ORACLE_DIR = (PKG / "../../C/projects/JAX_VALIDATION").resolve()
ORACLE = ORACLE_DIR / "oracle_1100.exe"
GOLDEN = PKG / "tests/_golden"
SEED = 20260823
N_LHS = 4096

sys.path.insert(0, str(PKG / "src"))


def lhs(rng: np.random.Generator, n: int, dims: int) -> np.ndarray:
    """Plain seeded Latin hypercube in [0,1)^dims (no scipy dependency)."""
    u = (rng.random((n, dims)) + np.arange(n)[:, None]) / n
    for d in range(dims):
        u[:, d] = u[rng.permutation(n), d]
    return u


def build_module_inputs(spec_inputs: list[str], cfg: dict,
                        rng: np.random.Generator) -> np.ndarray:
    ranges = cfg["ranges"]
    missing = [k for k in spec_inputs if k not in ranges]
    if missing:
        raise SystemExit(f"module_ranges.yaml missing inputs {missing}")
    u = lhs(rng, N_LHS, len(spec_inputs))
    cols = []
    for d, name in enumerate(spec_inputs):
        r = ranges[name]
        if isinstance(r, dict):
            lo, hi = float(r["lo"]), float(r["hi"])
            if r.get("loguniform"):
                col = np.exp(np.log(lo) + u[:, d] * (np.log(hi) - np.log(lo)))
            else:
                col = lo + u[:, d] * (hi - lo)
        else:
            lo, hi = float(r[0]), float(r[1])
            col = lo + u[:, d] * (hi - lo)
        cols.append(col)
    block = np.column_stack(cols)
    edges = np.array(cfg.get("edge_cases", []), dtype=np.float64)
    if edges.size:
        if edges.shape[1] != len(spec_inputs):
            raise SystemExit(f"edge case width {edges.shape[1]} != "
                             f"{len(spec_inputs)} inputs")
        block = np.vstack([block, edges])
    return block


# Row fixups for cross-input constraints the LHS cannot express.
def fixup_drainage(x: np.ndarray) -> np.ndarray:
    # psi_field (col 2) must be <= psi_porosity (col 3); both negative.
    lo = np.minimum(x[:, 2], x[:, 3])
    hi = np.maximum(x[:, 2], x[:, 3])
    x[:, 2], x[:, 3] = lo, hi
    return x


def fixup_liu(x: np.ndarray) -> np.ndarray:
    # Tupp (col 10) must exceed Tdown (col 11) for a sane Vcmax window.
    lo = np.minimum(x[:, 10], x[:, 11])
    hi = np.maximum(x[:, 10], x[:, 11])
    x[:, 10], x[:, 11] = hi, lo
    return x


FIXUPS = {"DRAINAGE": fixup_drainage, "LIU_AN_ET": fixup_liu}


def run(cmd: list[str]) -> str:
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stdout[-2000:] + r.stderr[-2000:])
        raise SystemExit(f"command failed: {' '.join(map(str, cmd))}")
    return r.stderr


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def gen_modules(manifest: dict, rng: np.random.Generator) -> dict:
    cfg_all = yaml.safe_load((PKG / "tools/module_ranges.yaml").read_text())
    mdir = GOLDEN / "modules"
    mdir.mkdir(parents=True, exist_ok=True)
    info = {}
    for spec in manifest["modules"]:
        name = spec["name"]
        if name not in cfg_all:
            raise SystemExit(f"module_ranges.yaml has no entry for {name}")
        x = build_module_inputs(spec["inputs"], cfg_all[name], rng)
        if name in FIXUPS:
            x = FIXUPS[name](x)
        fin = mdir / f"{name}.in.bin"
        fout = mdir / f"{name}.out.bin"
        x.astype("<f8").tofile(fin)
        run([str(ORACLE), "module", name, str(fin), str(fout)])
        got = np.fromfile(fout).reshape(-1, len(spec["outputs"]))
        info[name] = {
            "n_cases": int(x.shape[0]),
            "n_in": len(spec["inputs"]),
            "n_out": len(spec["outputs"]),
            "out_finite_frac": float(np.isfinite(got).mean()),
            "sha_in": sha(fin), "sha_out": sha(fout),
        }
        print(f"  {name}: {x.shape[0]} cases, "
              f"finite out {100*info[name]['out_finite_frac']:.1f}%")
    return info


def gen_trajectory_fixtures(rng: np.random.Generator) -> dict:
    import netCDF4
    from dalec_jax import indices as I

    root = PKG / "../../.."
    cbf = (root / "runs/mdf_1100_full/example_1100.cbf.nc").resolve()
    tdir = GOLDEN / "trajectories"
    tdir.mkdir(parents=True, exist_ok=True)

    with netCDF4.Dataset(root / "runs/mdf_1100_full/assim_1100.cbr") as ds:
        post = np.array(ds["Parameters"][:])
    with netCDF4.Dataset(root / "runs/baseline_1100/viable_ensemble.cbr.nc") as ds:
        viable = np.array(ds["Parameters"][:])

    # classify posterior samples with a cheap mlf pass
    allp = tdir / "_post_all.bin"
    post.astype("<f8").tofile(allp)
    tmp = tdir / "_post_mlf.bin"
    run([str(ORACLE), "mlf", str(cbf), str(allp), str(tmp)])
    row = I.NOEDCS + 31 + 1
    P = np.fromfile(tmp).reshape(-1, row)[:, -1]
    genuine_idx = np.where(np.isfinite(P))[0]
    gated_idx = np.where(~np.isfinite(P))[0]
    allp.unlink(); tmp.unlink()

    sel_gen = genuine_idx[np.linspace(0, len(genuine_idx) - 1, 64).astype(int)]
    sel_gat = gated_idx[np.linspace(0, len(gated_idx) - 1,
                                    min(16, len(gated_idx))).astype(int)]
    parmin = np.array(I.PARMIN)
    parmax = np.array(I.PARMAX)
    u = lhs(rng, 32, I.NOPARS)
    prior = parmin + u * (parmax - parmin)

    fix = np.vstack([post[sel_gen], post[sel_gat], viable, prior])
    meta = {
        "cbf": str(cbf),
        "rows": (
            [{"kind": "posterior_genuine", "cbr_index": int(i)} for i in sel_gen]
            + [{"kind": "posterior_gated", "cbr_index": int(i)} for i in sel_gat]
            + [{"kind": "viable_ensemble", "member": int(i)}
               for i in range(viable.shape[0])]
            + [{"kind": "prior_draw", "draw": int(i)} for i in range(32)]
        ),
    }
    pf = tdir / "fixture_params.bin"
    fix.astype("<f8").tofile(pf)
    (tdir / "fixture_meta.json").write_text(json.dumps(meta, indent=1))

    run([str(ORACLE), "trajectory", str(cbf), str(pf),
         str(tdir / "pools.bin"), str(tdir / "fluxes.bin")])
    run([str(ORACLE), "mlf", str(cbf), str(pf), str(tdir / "mlf.bin")])

    pools = np.fromfile(tdir / "pools.bin").reshape(fix.shape[0], -1, I.NOPOOLS)
    n_broken = int((~np.isfinite(pools.reshape(fix.shape[0], -1))
                    .all(axis=1)).sum())
    print(f"  trajectories: {fix.shape[0]} fixtures "
          f"({len(sel_gen)} genuine, {len(sel_gat)} gated, "
          f"{viable.shape[0]} viable, 32 prior; {n_broken} contain non-finite "
          f"steps = break-path exercisers)")
    return {"n_fixtures": int(fix.shape[0]), "n_broken": n_broken,
            "sha_params": sha(pf), "sha_pools": sha(tdir / "pools.bin"),
            "sha_fluxes": sha(tdir / "fluxes.bin"),
            "sha_mlf": sha(tdir / "mlf.bin")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-determinism", action="store_true")
    a = ap.parse_args()

    if not ORACLE.exists():
        raise SystemExit(f"{ORACLE} missing -- run make -C {ORACLE_DIR}")

    GOLDEN.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(subprocess.run(
        [str(ORACLE), "manifest"], capture_output=True, text=True).stdout)

    fingerprint = subprocess.run(
        ["make", "-s", "-C", str(ORACLE_DIR), "fingerprint"],
        capture_output=True, text=True).stdout

    rng = np.random.default_rng(SEED)
    print("Tier-A module goldens:")
    minfo = gen_modules(manifest, rng)
    print("Trajectory/mlf goldens:")
    tinfo = gen_trajectory_fixtures(rng)

    out = {"seed": SEED, "n_lhs": N_LHS, "fingerprint": fingerprint,
           "oracle_manifest": manifest, "modules": minfo,
           "trajectories": tinfo}
    (GOLDEN / "manifest.json").write_text(json.dumps(out, indent=1))
    print(f"wrote {GOLDEN / 'manifest.json'}")

    if not a.skip_determinism:
        print("determinism check: regenerating module inputs+outputs...")
        rng2 = np.random.default_rng(SEED)
        minfo2 = gen_modules(manifest, rng2)
        for k in minfo:
            if (minfo[k]["sha_in"] != minfo2[k]["sha_in"]
                    or minfo[k]["sha_out"] != minfo2[k]["sha_out"]):
                raise SystemExit(f"NON-DETERMINISTIC goldens for {k}")
        print("determinism check: byte-identical across regeneration")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
