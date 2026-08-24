"""Second figure: where the A100 catches up, and what a node is worth.

Panel A -- per-member gradient cost vs vmap batch size on one A100, against the
measured Julia 1-core cost. Because the GPU is latency-bound (total time is
nearly flat in B), per-member cost falls ~linearly and the two curves cross.
Panel B -- throughput, gradients/s, measured on every configuration tested.
"""
import json, os, glob
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch

R = os.path.join(os.path.dirname(__file__), "..", "results")
FIGDIR = os.path.join(os.path.dirname(__file__), "..", "figures")
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, INK3, SURF = "#0b0b0b", "#52514e", "#8a8880", "#fcfcfb"
L = lambda n: json.load(open(os.path.join(R, n)))

gb = L("jax_gpu_batched.json"); jc = L("jax_cpu.json"); jl1 = L("julia_cpu.json")
jlb = {int(os.path.basename(p).split("_t")[1].split(".")[0]): json.load(open(p))
       for p in glob.glob(os.path.join(R, "julia_batched_t*.json"))}
CFG = "default"
B = sorted(int(b) for b in gb["configs"][CFG])
gpu_pm = [gb["configs"][CFG][str(b)]["per_member_ms"] for b in B]
gpu_tot = [gb["configs"][CFG][str(b)]["total_ms"] for b in B]
jl_1core = jlb[1]["configs"][CFG]["per_member_ms"]

fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.6, 4.9),
                               gridspec_kw={"width_ratios": [1, 1.25]})
fig.patch.set_facecolor(SURF)

# ---------------- Panel A: crossover ----------------
axA.set_facecolor(SURF)
axA.plot(B, gpu_pm, "-o", color=S3, lw=2, ms=8, mec=SURF, mew=2, zorder=4,
         label="JAX, 1$\\times$A100, vmap batch $B$")
axA.axhline(jl_1core, color=S1, lw=2, ls=(0, (5, 3)), zorder=3,
            label="Julia, 1 CPU core (measured)")
axA.set_xscale("log"); axA.set_yscale("log")
axA.set_xticks(B); axA.set_xticklabels([str(b) for b in B])
axA.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
axA.set_xlabel("vmap batch size $B$ (independent restarts)", fontsize=9.5, color=INK2)
axA.set_ylabel("ms per gradient, per member  (log)", fontsize=9.5, color=INK2)
axA.set_title("A100 total time is nearly flat in $B$ (%.0f$\\to$%.0f ms for 1$\\to$512),\n"
              "so per-member cost falls until it crosses one CPU core at $B\\approx$%d"
              % (gpu_tot[0], gpu_tot[-1], round(np.mean(gpu_tot) / jl_1core)),
              fontsize=9.3, color=INK, loc="left", pad=9)
xc = np.mean(gpu_tot) / jl_1core
axA.plot([xc], [jl_1core], "o", ms=9, mfc="none", mec=INK2, mew=1.6, zorder=6)
axA.annotate("crossover\n$B\\approx$%d" % round(xc), (xc, jl_1core),
             textcoords="offset points", xytext=(-6, 26), ha="center",
             fontsize=8.4, color=INK2)
for b, v in zip(B, gpu_pm):
    axA.annotate(f"{v:.2f}" if v < 10 else f"{v:.0f}", (b, v), textcoords="offset points",
                 xytext=(7, 7), fontsize=8.2, color=INK3)
axA.legend(frameon=False, fontsize=8.8, labelcolor=INK2, loc="lower left",
           bbox_to_anchor=(0.0, 0.0))

# ---------------- Panel B: throughput ----------------
ROWS = [("JAX, 1 CPU core",              1000.0 / jc["configs"][CFG]["grad_ms"],  S2),
        ("Julia, 1 CPU core",            1000.0 / jl_1core,                        S1),
        ("JAX, 1$\\times$A100 ($B$=512)", 512.0 / (gpu_tot[-1] / 1000.0) / 1000.0 * 1000.0, S3),
        ("Julia, 8 cores",               1000.0 / jlb[8]["configs"][CFG]["per_member_ms"],   S1),
        ("Julia, 32 cores",              1000.0 / jlb[32]["configs"][CFG]["per_member_ms"],  S1),
        ("Julia, 128 cores",             1000.0 / jlb[128]["configs"][CFG]["per_member_ms"], S1)]
ROWS.sort(key=lambda r: r[1])
axB.set_facecolor(SURF)
ymaxB = max(r[1] for r in ROWS) * 1.30
for i, (lbl, v, col) in enumerate(ROWS):
    p0 = axB.transData.transform((0, i - 0.31)); p1 = axB.transData.transform((v, i + 0.31))
    r = min(4 * fig.dpi / 72.0, abs(p1[0] - p0[0]), abs(p1[1] - p0[1]) / 2)
    vs = [(p0[0], p0[1]), (p1[0] - r, p0[1]), (p1[0], p0[1]), (p1[0], p0[1] + r),
          (p1[0], p1[1] - r), (p1[0], p1[1]), (p1[0] - r, p1[1]), (p0[0], p1[1]), (p0[0], p0[1])]
    cd = [Path.MOVETO, Path.LINETO, Path.CURVE3, Path.CURVE3,
          Path.LINETO, Path.CURVE3, Path.CURVE3, Path.LINETO, Path.CLOSEPOLY]
    inv = axB.transData.inverted()
    axB.add_patch(PathPatch(Path([inv.transform(v_) for v_ in vs], cd),
                            facecolor=col, edgecolor="none", zorder=3))
    axB.text(v + ymaxB * 0.015, i, f"{v:,.0f}", va="center", ha="left",
             fontsize=9, color=INK2)
axB.set_yticks(range(len(ROWS)))
axB.set_yticklabels([r[0] for r in ROWS], fontsize=9.3, color=INK2)
axB.set_xlim(0, ymaxB); axB.set_ylim(-0.6, len(ROWS) - 0.4)
axB.set_xlabel("gradients per second (config 2, all measured)", fontsize=9.5, color=INK2)
axB.set_title("One A100 is worth ~2.5 CPU cores on this model:\n"
              "tiny sequential state never lets the GPU use its FLOPs",
              fontsize=9.3, color=INK, loc="left", pad=9)
axB.grid(axis="x", color="#e6e5e0", lw=0.8, zorder=0); axB.set_axisbelow(True)

for ax in (axA, axB):
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#d8d7d1"); ax.spines["left"].set_color("#d8d7d1")
    ax.tick_params(axis="both", length=0, colors=INK3)
axB.spines["left"].set_visible(False)
axA.grid(True, which="major", color="#e6e5e0", lw=0.8, zorder=0); axA.set_axisbelow(True)

fig.text(0.006, 0.975, "DifferLand DALEC990: the batched regime — where does a GPU pay off?",
         fontsize=12, color=INK, ha="left", va="top")
fig.text(0.006, 0.014,
         "US-Var, T=3,287, float32, config 2 (β-JS). Batch members are independently perturbed so nothing is shared across the batch. "
         "Julia multicore = one (32,T) buffer pair per chunk,\nB=512 split evenly; at 128 threads only 4 members per thread, so that point "
         "under-feeds the cores and is a lower bound on Julia's scaling. Compile time excluded throughout.",
         fontsize=7.5, color=INK3, va="bottom")
fig.subplots_adjust(left=0.083, right=0.985, top=0.775, bottom=0.175, wspace=0.30)
os.makedirs(FIGDIR, exist_ok=True)
out = os.path.join(FIGDIR, "differland_batch_scaling.png")
fig.savefig(out, dpi=190, facecolor=SURF); print("wrote", out)
