"""Figure for the Julia-vs-JAX DifferLand comparison.

Form: small multiples (one panel per model config), grouped bars, log y.
The measures span ~2 orders of magnitude across backends, so log is the honest
axis; every bar carries a direct value label (also discharges the contrast-relief
rule for the aqua slot).
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch

R = os.path.join(os.path.dirname(__file__), "..", "results")
FIGDIR = os.path.join(os.path.dirname(__file__), "..", "figures")
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"      # blue, orange, aqua
INK, INK2, INK3 = "#0b0b0b", "#52514e", "#8a8880"
SURF = "#fcfcfb"

jl = json.load(open(os.path.join(R, "julia_cpu.json")))
jc = json.load(open(os.path.join(R, "jax_cpu.json")))
gp = os.path.join(R, "jax_gpu.json")
jg = json.load(open(gp)) if os.path.exists(gp) else None
if jg is not None and len(jg.get("configs", {})) < 2:
    print("jax_gpu.json is partial (%d configs) -- omitting the GPU series"
          % len(jg.get("configs", {})))
    jg = None

SERIES = [("Julia, 1 CPU core\n(Enzyme reverse)", S1, jl, "julia"),
          ("JAX, 1 CPU core\n(XLA)", S2, jc, "jax"),
          ("JAX, 1x A100\n(XLA)", S3, jg, "jax")]
SERIES = [s for s in SERIES if s[2] is not None]
MEASURES = [("forward\nmodel", "forward_ms"), ("value +\ngradient", "grad_ms"),
            ("Adam\niteration", None)]
CONFIGS = [("default", "config 2: $\\beta$-JS water stress\n42 trainable parameters"),
           ("nn_whole", "config 5: GPP&ET from an MLP\n240 trainable parameters")]

def get(d, kind, cfg, key):
    c = d["configs"][cfg]
    if key is None:
        return c["adam_iter_ms"] if kind == "julia" else c["adam_iter_ms_sync"]
    return c[key]

def rounded_bar(ax, x, w, h, color, r_pt=4):
    """Bar anchored to the baseline with rounded data-end (top) corners."""
    ylo = ax.get_ylim()[0]
    # radius in data units, from points, on a log axis -> work in display space
    p0 = ax.transData.transform((x, ylo)); p1 = ax.transData.transform((x + w, h))
    xr0, yr0 = p0; xr1, yr1 = p1
    r = min(r_pt * ax.figure.dpi / 72.0, abs(xr1 - xr0) / 2, abs(yr1 - yr0))
    verts = [(xr0, yr0), (xr0, yr1 - r), (xr0, yr1), (xr0 + r, yr1),
             (xr1 - r, yr1), (xr1, yr1), (xr1, yr1 - r), (xr1, yr0), (xr0, yr0)]
    codes = [Path.MOVETO, Path.LINETO, Path.CURVE3, Path.CURVE3,
             Path.LINETO, Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CLOSEPOLY]
    inv = ax.transData.inverted()
    ax.add_patch(PathPatch(Path([inv.transform(v) for v in verts], codes),
                           facecolor=color, edgecolor="none", zorder=3, clip_on=False))

fig, axes = plt.subplots(1, len(CONFIGS), figsize=(11.6, 5.15), sharey=True)
fig.patch.set_facecolor(SURF)
vals_all = [get(d, kind, cfg, key) for _, _, d, kind in SERIES
            for cfg, _ in CONFIGS for _, key in MEASURES]
# log only when the data actually spans a decade or more; otherwise linear reads better
LOG = max(vals_all) / min(vals_all) > 20
ymax = max(vals_all) * (2.6 if LOG else 1.22)
ymin = min(vals_all) / 2.2 if LOG else 0.0

for ax, (cfg, title) in zip(axes, CONFIGS):
    ax.set_facecolor(SURF)
    if LOG:
        ax.set_yscale("log")
        ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_ylim(ymin, ymax)
    n = len(SERIES); group_w = 0.74; bw = group_w / n
    gap = 0.012                                        # 2px surface gap between bars
    for gi, (_, key) in enumerate(MEASURES):
        for si, (_, color, d, kind) in enumerate(SERIES):
            v = get(d, kind, cfg, key)
            x = gi - group_w / 2 + si * bw + gap / 2
            rounded_bar(ax, x, bw - gap, v, color)
            ax.text(x + (bw - gap) / 2, v * 1.06 if LOG else v + ymax * 0.022,
                    f"{v:.2f}" if v < 100 else f"{v:.0f}",
                    ha="center", va="bottom", fontsize=8.5, color=INK2)
    ax.set_xlim(-0.52, len(MEASURES) - 0.48)
    ax.set_xticks(range(len(MEASURES)))
    ax.set_xticklabels([m for m, _ in MEASURES], fontsize=9.5, color=INK2)
    ax.set_title(title, fontsize=9.8, color=INK, pad=7, loc="left")
    ax.grid(axis="y", color="#e6e5e0", lw=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#d8d7d1")
    ax.tick_params(axis="both", length=0, colors=INK3)

axes[0].set_ylabel("wall time per call, ms" + ("  (log scale)" if LOG else ""),
                   fontsize=9.5, color=INK2, labelpad=8)
handles = [plt.Line2D([], [], marker="s", ls="", ms=9, color=c, label=lbl)
           for lbl, c, _, _ in SERIES]
fig.legend(handles=handles, loc="upper left", ncol=len(SERIES), frameon=False,
           fontsize=9, labelcolor=INK2, bbox_to_anchor=(0.006, 0.945),
           handletextpad=0.55, columnspacing=2.6, borderaxespad=0)
fig.text(0.008, 0.978,
         "DifferLand DALEC990: same model, same float32, same 3,287-step site — lower is better",
         fontsize=12, color=INK, ha="left", va="top")
fig.text(0.008, 0.012,
         "US-Var, T=3,287 daily steps. Julia/JAX-CPU pinned to one core with taskset; JIT and Enzyme compile "
         "time excluded (reported separately in the table).\n"
         "Adam iteration reads the loss every step in both languages, which forces a device sync on the JAX side "
         "— that is what the upstream calibration loop does.",
         fontsize=7.6, color=INK3, va="bottom")
fig.subplots_adjust(left=0.075, right=0.985, top=0.755, bottom=0.165, wspace=0.07)
os.makedirs(FIGDIR, exist_ok=True)
out = os.path.join(FIGDIR, "differland_julia_vs_jax.png")
fig.savefig(out, dpi=190, facecolor=SURF)
print("wrote", out)
