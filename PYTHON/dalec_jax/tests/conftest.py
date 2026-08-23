"""Equivalence-suite environment: must run BEFORE jax is imported anywhere.

Two mandatory settings (see tests/TOLERANCES.md "Fusion findings"):
- XLA's algebraic simplifier rewrites x/c -> x*(1/c) and (x/c1)*c2 ->
  x*(c2/c1), changing rounding vs the -O0 C reference. Disabling the
  `algsimp` HLO pass restores bit-identical arithmetic.
- The login shell's LD_LIBRARY_PATH (/usr/local/cuda/lib64) breaks the CUDA
  plugin; equivalence tests are CPU-only by policy anyway.
"""
import os
import sys

_FLAG = "--xla_disable_hlo_passes=algsimp"

if "jax" in sys.modules:
    raise RuntimeError(
        "conftest.py must configure XLA_FLAGS before jax is imported; "
        "something imported jax at collection time")

flags = os.environ.get("XLA_FLAGS", "")
if "xla_disable_hlo_passes" not in flags:
    os.environ["XLA_FLAGS"] = (flags + " " + _FLAG).strip()
elif "algsimp" not in flags:
    raise RuntimeError(
        f"XLA_FLAGS already sets xla_disable_hlo_passes without algsimp "
        f"({flags!r}); refusing to run equivalence tests")

os.environ.setdefault("JAX_PLATFORMS", "cpu")
