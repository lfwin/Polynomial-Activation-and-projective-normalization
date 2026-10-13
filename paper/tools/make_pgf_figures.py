#!/usr/bin/env python3
"""生成论文全部 10 张图的纯矢量 PGF（零光栅、可内联）。"""
import sys, math
from pathlib import Path

import matplotlib
matplotlib.use("pgf")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors

plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.titlesize": 10,
    "axes.labelsize": 9, "legend.fontsize": 7.5, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "axes.grid": True, "grid.alpha": 0.25,
    "grid.linewidth": 0.5, "pgf.texsystem": "pdflatex", "pgf.rcfonts": False,
    "pgf.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
    "savefig.bbox": "tight",
})

sys.path.insert(0, "/home/lantuo/Documents/NN_paper/projective-poly-mnist/experiments")
import analyze

OUT = Path("/tmp/pgf_out"); OUT.mkdir(exist_ok=True)
recs = analyze.load_all("/home/lantuo/Documents/NN_paper/projective-poly-mnist/experiments")
print("记录:", len(recs))

_orig = plt.Figure.savefig
def savefig_pgf(self, fname, *a, **kw):
    p = Path(fname)
    if p.suffix == ".pdf":
        kw.pop("dpi", None)
        return _orig(self, str(OUT / (p.stem + ".pgf")), *a, **kw)
    return _orig(self, fname, *a, **kw)
plt.Figure.savefig = savefig_pgf


def draw_colorbar(ax, cmap_name, vmin, vmax, label, n=64, fmt="{:.0f}",
                  nticks=5, fontsize=7.5):
    cmap = plt.get_cmap(cmap_name)
    fig = ax.figure
    bb = ax.get_position()
    cax = fig.add_axes([bb.x1 + 0.020, bb.y0, 0.018, bb.height])
    ys = np.linspace(0, 1, n + 1)
    for k in range(n):
        cax.add_patch(Rectangle((0, ys[k]), 1, ys[1] - ys[0],
                                facecolor=cmap(k / (n - 1)), edgecolor="none"))
    cax.set_xlim(0, 1); cax.set_ylim(0, 1)
    cax.set_xticks([]); cax.yaxis.tick_right()
    ticks = np.linspace(vmin, vmax, nticks)
    cax.set_yticks([(t - vmin) / (vmax - vmin) for t in ticks])
    cax.set_yticklabels([fmt.format(t) for t in ticks], fontsize=fontsize)
    cax.set_ylabel(label, fontsize=fontsize, rotation=90, labelpad=4)
    cax.tick_params(length=2, width=0.5)
    cax.spines["left"].set_visible(False)
    cax.spines["top"].set_visible(False)
    return cax


def vector_heatmap(M, cmap_name, vmin, vmax, xticklabels, yticklabels,
                   figsize, cbar_label, rotation=0, fs=6.5):
    cmap = plt.get_cmap(cmap_name)
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    fig, ax = plt.subplots(figsize=figsize)
    rows, cols = M.shape
    for i in range(rows):
        for j in range(cols):
            v = M[i, j]
            fc = cmap(norm(v)) if not math.isnan(v) else (1, 1, 1, 0)
            ax.add_patch(Rectangle((j, i), 1, 1, facecolor=fc, edgecolor="none"))
            if not math.isnan(v):
                ax.text(j + 0.5, i + 0.5, f"{v:.0f}", ha="center", va="center",
                        fontsize=fs, color="k" if 0.30 < norm(v) < 0.88 else "w")
    ax.set_xlim(0, cols); ax.set_ylim(0, rows)
    ax.set_xticks(np.arange(cols) + 0.5)
    ax.set_xticklabels(xticklabels, rotation=rotation,
                       ha="right" if rotation else "center")
    ax.set_yticks(np.arange(rows) + 0.5)
    ax.set_yticklabels(yticklabels)
    ax.invert_yaxis(); ax.grid(False)
    draw_colorbar(ax, cmap_name, vmin, vmax, cbar_label)
    return fig, ax


def fig_degree_vec(recs, out_dir):
    degs, lams = [2, 3, 4, 5], [0.001, 0.01, 0.05, 0.1]
    M = np.full((len(degs), len(lams)), np.nan)
    for i, d in enumerate(degs):
        for j, lm in enumerate(lams):
            r = analyze.one(recs, dataset="mnist", activation="poly", degree=d,
                            lam=lm, norm="l2", depth=10, hidden=128, seed=42)
            if r:
                M[i, j] = r["test_accuracy"] * 100
    fig, ax = vector_heatmap(M, "viridis", np.nanmin(M), np.nanmax(M),
                             [f"{l:g}" for l in lams], [f"$d={d}$" for d in degs],
                             (4.8, 2.6), "Test accuracy (%)", fs=8)
    ax.set_xlabel(r"$\lambda$")
    fig.savefig(OUT / "fig_degree.pdf"); plt.close(fig)


def fig_synth_vec(recs, out_dir):
    names = [("rings", "Rings"), ("spiral", "Spiral"), ("blobs", "Blobs"),
             ("xor", "XOR"), ("checker", "Checker"), ("cubes", "3-bit XOR"),
             ("parity", "Parity"), ("gauss_highdim", "High-dim")]
    acts = ["identity", "relu", "gelu", "silu", "tanh", "poly", "swiglu", "geglu"]
    M = np.full((len(acts), len(names)), np.nan)
    for j, (name, _) in enumerate(names):
        for i, a in enumerate(acts):
            r = analyze.one(recs, dataset="synth_train", synth_name=name,
                            activation=a, depth=3, hidden=64, seed=0)
            if r:
                M[i, j] = r["test_accuracy"] * 100
    fig, ax = vector_heatmap(
        M, "RdYlGn", 50, 100, [l for _, l in names],
        [r"Poly+$\ell_2$" if a == "poly" else analyze.NICE.get(a, a) for a in acts],
        (6.8, 3.3), "Test accuracy (%)", rotation=35, fs=6.5)
    fig.savefig(OUT / "fig_synth.pdf"); plt.close(fig)


analyze.fig_degree_heatmap = fig_degree_vec
analyze.fig_synth = fig_synth_vec

for name in ["fig_activation", "fig_norm", "fig_degree_heatmap", "fig_depth",
             "fig_act_norm", "fig_synth", "fig_synth_depth", "fig_poly_degree",
             "fig_glu_norm", "fig_width"]:
    try:
        getattr(analyze, name)(recs, str(OUT)); print("OK  ", name)
    except Exception as e:
        print("FAIL", name, type(e).__name__, e)

print("\n=== 光栅检查 ===")
bad = []
for f in sorted(OUT.glob("*.pgf")):
    t = f.read_text()
    has = ("pgfimage" in t) or (".png" in t) or (".jpg" in t)
    if has:
        bad.append(f.name)
    print(f"  {f.name:26s} {f.stat().st_size:7d} B  光栅={has}")
print("含光栅:", bad or "无（全部纯矢量）")
