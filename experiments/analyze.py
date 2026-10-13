#!/usr/bin/env python3
"""
汇总所有实验 JSON，生成论文用的表格（LaTeX 片段）与图（PDF）。

用法:
  python experiments/analyze.py --exp-dir experiments --paper-dir paper
"""
import argparse
import json
import glob
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif"],
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 140,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linewidth": 0.5,
})

# 激活集合必须与 experiments/run_all.py 一致
ACTS = ["identity", "tanh", "relu", "gelu", "silu",
        "poly", "poly_proj",
        "swiglu", "geglu", "polyglu"]

GATED = ["swiglu", "geglu", "polyglu"]
GATED_H = 82
ELEM_H = 128

NICE = {
    "identity": "Identity", "tanh": "Tanh",
    "relu": "ReLU", "gelu": "GELU", "silu": "SiLU",
    "poly": r"Poly $z+\lambda z^3$",
    "poly_proj": r"Poly + $\ell_2$ proj.", "swiglu": "SwiGLU", "geglu": "GeGLU",
    "polyglu": r"Poly-GLU",
    "legendre": r"Legendre (learned)", "chebyshev": r"Chebyshev (learned)",
    "bspline": r"B-spline (learned)",
    "none": "None", "batch": "BatchNorm", "layer": "LayerNorm",
    "rms": "RMSNorm", "l2": r"Projective $\ell_2$",
}


def hidden_of(act):
    """门控变体用 H=82 与逐元素 H=128 参数量对齐。"""
    return GATED_H if act in GATED else ELEM_H



# ============================================================
# 载入
# ============================================================

def load_all(exp_dir):
    recs = []
    for f in glob.glob(str(Path(exp_dir) / "*" / "*.json")):
        with open(f, encoding="utf-8") as fh:
            try:
                r = json.load(fh)
                r["_path"] = f
                recs.append(r)
            except Exception:
                pass
    return recs


def key_of(r):
    a = r["args"]
    return (a.get("dataset"), a.get("synth_name"), a.get("activation"),
            a.get("degree"), a.get("lam"), a.get("norm"), a.get("norm_pos"),
            a.get("depth"), a.get("hidden"), a.get("seed"))


# 独立参考组：与主网格同配置但 epoch 预算不同（ref_orig=20ep, ref_fidelity=2ep）。
# 它们共用同一个配置 key，混进分析总体会让未过滤 epochs 的查询取到错误记录
# （\refTestAcc 会从 20ep 的 96.95 掉回 10ep 的 95.93）。默认排除，需要时用 out=。
_SOLO_GROUPS = {"ref_orig", "ref_fidelity"}


def _is_solo(r):
    p = r.get("_path")
    return p is not None and Path(p).parent.name in _SOLO_GROUPS


def index(recs, **q):
    # out="width_mnist" 之类的限定：按结果目录名筛选，避免不同实验组的
    # 同名配置互相污染（例如 degree_mnist 里 19 条 poly/l2 会把
    # width_mnist 的宽度消融查询顶掉，产出根本不存在于该组的幻值）。
    # 未指定 out 时，默认排除独立参考组（见上）。
    out_dir = q.pop("out", None)
    # 归一化位置默认只取 post：norm_mnist 组里 layer/rms 同时有 pre 与 post
    # 两条记录，one() 取 m[0]，不钉死位置就会随 glob 顺序在 post/pre 之间
    # 抖动（96.29 vs 95.09）。需要 pre 时显式传 norm_pos="pre"。
    if "norm_pos" not in q:
        q["norm_pos"] = "post"
    out = []
    for r in recs:
        a = r["args"]
        if out_dir is not None:
            p = r.get("_path")
            if p is None or Path(p).parent.name != out_dir:
                continue
        elif _is_solo(r):
            continue
        ok = True
        for k, v in q.items():
            if k == "dataset_name":
                if a.get("dataset") not in ("synth_train", v):
                    ok = False
            elif a.get(k) != v:
                ok = False
        if ok:
            out.append(r)
    # 稳定排序：消除 glob/文件系统顺序带来的不确定性，同名配置的多个记录
    # （多 seed、多 epoch）按确定性顺序排列，one() 结果可复现。
    out.sort(key=lambda r: (r["_path"] or ""))
    return out


def one(recs, **q):
    m = index(recs, **q)
    if not m:
        return None
    return m[0]


def agg(vals):
    """返回 (mean, std)。"""
    if not vals:
        return (float("nan"), float("nan"))
    return (float(np.mean(vals)), float(np.std(vals)))


# ============================================================
# 表格
# ============================================================

def fmt(x, d=2):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "--"
    return f"{x:.{d}f}"


def table_activation(recs, out_dir):
    """表 1：MNIST 10 层，各激活函数的性能 / 参数量 / 单 epoch 耗时。"""
    rows = []
    for a in ACTS:
        r = one(recs, dataset="mnist", activation=a, depth=10,
                hidden=hidden_of(a), seed=42, tag=f"mnist-L10")
        if r is None:
            continue
        rows.append((a, r))
    lines = []
    L = lines.append
    L(r"\begin{table}[t]")
    L(r"\centering")
    L(r"\caption{Activations on MNIST at depth 10 (128 hidden units, AdamW, "
      r"10 epochs, seed 42, CPU, no normalization; gated variants use "
      r"hidden width 82 for parameter parity).}")
    L(r"\label{tab:activation}")
    L(r"\small")
    L(r"\begin{tabular}{lrrrrr}")
    L(r"\toprule")
    L(r"Activation & Params & Train acc.\ & Test acc.\ & Test loss & s/epoch \\")
    L(r"\midrule")
    # 按测试准确率排序
    rows_sorted = sorted(rows, key=lambda z: -z[1]["test_accuracy"])
    best = rows_sorted[0][1]["test_accuracy"] if rows_sorted else 1
    for a, r in rows_sorted:
        acc = r["test_accuracy"]
        tra = r["train_accuracy"]
        fmtf = (lambda v: r"\textbf{" + v + "}") if abs(acc - best) < 1e-9 \
            else (lambda v: v)
        sp = r["seconds_per_epoch"]
        L(f"{NICE.get(a, a)} & {r['n_params']:,} & "
          f"{fmtf(f'{tra*100:.2f}')} & {fmtf(f'{acc*100:.2f}')} & "
          f"{r['test_loss']:.4f} & {sp:.1f} \\\\")
    L(r"\bottomrule")
    L(r"\end{tabular}")
    L(r"\end{table}")
    (Path(out_dir) / "tab_activation.tex").write_text("\n".join(lines),
                                                      encoding="utf-8")
    return rows


def table_norm(recs, out_dir):
    """表 2：归一化方式对比（ReLU 与 Poly 各一组，MNIST 10 层）。"""
    lines = []
    L = lines.append
    L(r"\begin{table}[t]")
    L(r"\centering")
    L(r"\caption{Normalization schemes on MNIST at depth 10, with ReLU and "
      r"with the polynomial activation $f(z)=z+\lambda z^3$. Pre/Post is the "
      r"placement of the normalization relative to the activation.}")
    L(r"\label{tab:norm}")
    L(r"\small")
    L(r"\begin{tabular}{llrrr}")
    L(r"\toprule")
    L(r"Activation & Norm & Pos & Test acc.\ (\%) & Worst-layer grad.\ ratio \\")
    L(r"\midrule")
    for act, tag in [("relu", "mnist-L10-relu"), ("poly", "mnist-L10-poly-all")]:
        combos = [("none", "post"), ("batch", "post"), ("layer", "post"),
                  ("rms", "post"), ("layer", "pre"), ("rms", "pre"),
                  ("l2", "post")]
        for norm, pos in combos:
            r = one(recs, dataset="mnist", activation=act, norm=norm,
                    depth=10, hidden=128, seed=42, tag=tag)
            if r is None:
                continue
            ratio = grad_ratio(r)
            L(f"{NICE.get(act, act)} & {NICE.get(norm, norm)} & {pos} & "
              f"{r['test_accuracy']*100:.2f} & {fmt(ratio, 4)} \\\\")
        L(r"\midrule")
    L(r"\bottomrule")
    L(r"\end{tabular}")
    L(r"\end{table}")
    (Path(out_dir) / "tab_norm.tex").write_text("\n".join(lines),
                                                encoding="utf-8")


def grad_ratio(r):
    """首层梯度范数 / 末层梯度范数（训练结束后），衡量梯度传播是否均匀。"""
    g0 = r.get("grad_norm_end") or []
    if len(g0) < 2:
        return None
    if g0[-1] <= 0:
        return None
    return g0[0] / g0[-1]


def table_degree(recs, out_dir):
    """表 3：多项式阶数 x lam 网格（MNIST，poly + 射影归一化）。"""
    degs, lams = [2, 3, 4, 5], [0.001, 0.01, 0.05, 0.1]
    grid = {}
    for d in degs:
        for lm in lams:
            r = one(recs, dataset="mnist", activation="poly", degree=d, lam=lm,
                    norm="l2", depth=10, hidden=128, seed=42)
            grid[(d, lm)] = r
    lines = []
    L = lines.append
    L(r"\begin{table}[t]")
    L(r"\centering")
    L(r"\caption{Grid over the polynomial degree $d$ and the coefficient "
      r"$\lambda$ (MNIST, depth 10, projective $\ell_2$ normalization, "
      r"test accuracy in \%, 10 epochs).}")
    L(r"\label{tab:degree}")
    L(r"\small")
    L(r"\begin{tabular}{r" + "r" * len(lams) + r"}")
    L(r"\toprule")
    L(r"$d$ \textbackslash{} $\lambda$ & " +
      " & ".join(f"{lm:g}" for lm in lams) + r" \\")
    L(r"\midrule")
    best_val = max((r["test_accuracy"] for r in grid.values() if r),
                   default=None)
    for d in degs:
        cells = []
        for lm in lams:
            r = grid[(d, lm)]
            if r is None:
                cells.append("--")
                continue
            v = r["test_accuracy"] * 100
            s = f"{v:.2f}"
            if best_val is not None and abs(r["test_accuracy"] - best_val) < 1e-9:
                s = r"\textbf{" + s + "}"
            cells.append(s)
        L(f"{d} & " + " & ".join(cells) + r" \\")
    L(r"\bottomrule")
    L(r"\end{tabular}")
    L(r"\end{table}")
    (Path(out_dir) / "tab_degree.tex").write_text("\n".join(lines),
                                                  encoding="utf-8")
    return grid


def table_depth(recs, out_dir):
    """表 4：深度扩展性（MNIST）。"""
    depths = [2, 3, 5, 10, 15, 20, 30, 50]
    cfgs = [("relu", "none", "ReLU", "relu"),
            ("relu", "batch", "ReLU+BN", "relu-bn"),
            ("relu", "layer", "ReLU+LN", "relu-ln"),
            ("poly", "none", "Poly", "poly"),
            ("poly", "l2", "Poly+$\\ell_2$", "polyproj"),
            ("poly", "layer", "Poly+LN", "poly-ln"),
            ("poly", "batch", "Poly+BN", "poly-bn")]
    data = defaultdict(dict)
    for act, norm, label, tag in cfgs:
        for d in depths:
            r = one(recs, dataset="mnist", activation=act, norm=norm, depth=d,
                    hidden=128, seed=42, tag=f"mnist-L{d}-{tag}")
            if r:
                data[label][d] = r
    labels = [c[2] for c in cfgs if data[c[2]]]
    lines = []
    L = lines.append
    L(r"\begin{table}[t]")
    L(r"\centering")
    L(r"\caption{Depth scaling: test accuracy (\%) as a function of the "
      r"number of linear layers $L$ (MNIST, 128 hidden units, 10 epochs, "
      r"seed 42). $L=30,50$ come from a separate group of runs.}")
    L(r"\label{tab:depth}")
    L(r"\scriptsize")
    L(r"\begin{tabular}{l" + "r" * len(depths) + r"}")
    L(r"\toprule")
    L(r"Configuration & " + " & ".join(str(d) for d in depths) + r" \\")
    L(r"\midrule")
    for label in labels:
        cells = []
        for d in depths:
            r = data[label].get(d)
            cells.append(f"{r['test_accuracy']*100:.2f}" if r else "--")
        L(f"{label} & " + " & ".join(cells) + r" \\")
    L(r"\bottomrule")
    L(r"\end{tabular}")
    L(r"\end{table}")
    (Path(out_dir) / "tab_depth.tex").write_text("\n".join(lines),
                                                 encoding="utf-8")
    return data


def table_synth(recs, out_dir):
    """表 5：自构数据 + 3 层网络，激活函数对比（测试准确率 %）。"""
    names = [("rings", "Rings"), ("spiral", "Spiral"), ("blobs", "Blobs"),
             ("xor", "XOR"), ("checker", "Checkerboard"), ("cubes", "3-bit XOR"),
             ("parity", "10-bit Parity"), ("gauss_highdim", "High-dim Gaussian")]
    acts = ["identity", "relu", "gelu", "silu", "tanh", "poly",
            "swiglu", "geglu"]
    data = defaultdict(dict)
    for name, label in names:
        for a in acts:
            r = one(recs, dataset="synth_train", synth_name=name,
                    activation=a, depth=3, hidden=64, seed=0)
            if r:
                data[label][a] = r["test_accuracy"] * 100
    labels = [l for _, l in names if data[l]]
    lines = []
    L = lines.append
    L(r"\begin{table}[t]")
    L(r"\centering")
    L(r"\caption{Activations on the synthetic datasets (3-layer MLP, 64 "
      r"hidden units, AdamW, 60 epochs, 4000 training samples, test accuracy "
      r"in \%, seed 0). Polynomial/projective configurations are annotated.}")
    L(r"\label{tab:synth}")
    L(r"\scriptsize")
    L(r"\begin{tabular}{l" + "r" * len(labels) + r"}")
    L(r"\toprule")
    L(r"Activation & " + " & ".join(labels) + r" \\")
    L(r"\midrule")
    for a in acts:
        vals = [data[l].get(a) for l in labels]
        if all(v is None for v in vals):
            continue
        cells = [(r"\textbf{" + f"{v:.1f}" + "}") if (v is not None and
                  abs(v - max(x for x in vals if x is not None)) < 1e-9)
                 else (f"{v:.1f}" if v is not None else "--") for v in vals]
        name = NICE.get(a, a)
        if a == "poly":
            name = r"Poly+$\ell_2$"
        L(f"{name} & " + " & ".join(cells) + r" \\")
    L(r"\bottomrule")
    L(r"\end{tabular}")
    L(r"\end{table}")
    (Path(out_dir) / "tab_synth.tex").write_text("\n".join(lines),
                                                 encoding="utf-8")
    return data


def table_synth_norm(recs, out_dir):
    """表 6：自构数据上归一化对比。"""
    names = [("rings", "Rings"), ("spiral", "Spiral"), ("checker", "Checker"),
             ("parity", "Parity")]
    cfgs = [("poly", "l2", r"Poly+$\ell_2$"), ("poly", "none", "Poly"),
            ("poly", "batch", "Poly+BN"), ("poly", "layer", "Poly+LN"),
            ("poly", "rms", "Poly+RMS"), ("relu", "none", "ReLU"),
            ("relu", "batch", "ReLU+BN"), ("relu", "layer", "ReLU+LN")]
    data = defaultdict(dict)
    for name, label in names:
        for act, norm, tag in cfgs:
            r = one(recs, dataset="synth_train", synth_name=name,
                    activation=act, norm=norm, depth=3, hidden=64, seed=0)
            if r:
                data[label][tag] = r["test_accuracy"] * 100
    labels = [l for _, l in names if data[l]]
    lines = []
    L = lines.append
    L(r"\begin{table}[t]")
    L(r"\centering")
    L(r"\caption{Effect of the normalization scheme on the synthetic "
      r"datasets (3 layers, 64 hidden units, 60 epochs, test accuracy in "
      r"\%).}")
    L(r"\label{tab:synthnorm}")
    L(r"\small")
    L(r"\begin{tabular}{l" + "r" * len(labels) + r"}")
    L(r"\toprule")
    L(r"Configuration & " + " & ".join(labels) + r" \\")
    L(r"\midrule")
    for _, _, tag in [(a, n, t) for a, n, t in cfgs]:
        vals = [data[l].get(tag) for l in labels]
        if all(v is None for v in vals):
            continue
        cells = [f"{v:.1f}" if v is not None else "--" for v in vals]
        L(f"{tag} & " + " & ".join(cells) + r" \\")
    L(r"\bottomrule")
    L(r"\end{tabular}")
    L(r"\end{table}")
    (Path(out_dir) / "tab_synthnorm.tex").write_text("\n".join(lines),
                                                     encoding="utf-8")
    return data


def table_seeds(recs, out_dir):
    """表 7：种子重复性（mean +- std）。"""
    cfgs = [("relu", "none", "ReLU"), ("gelu", "none", "GELU"),
            ("poly", "l2", r"Poly+$\ell_2$"), ("poly", "batch", "Poly+BN"),
            ("poly", "layer", "Poly+LN")]
    lines = []
    L = lines.append
    L(r"\begin{table}[t]")
    L(r"\centering")
    L(r"\caption{Seed-to-seed variability on MNIST at depth 10, over five "
      r"random seeds (mean $\pm$ standard deviation of the test accuracy in "
      r"\%).}")
    L(r"\label{tab:seeds}")
    L(r"\small")
    L(r"\begin{tabular}{lrrr}")
    L(r"\toprule")
    L(r"Configuration & Test acc.\ & Std.\ dev.\ & Train loss \\")
    L(r"\midrule")
    for act, norm, label in cfgs:
        rs = index(recs, dataset="mnist", activation=act, norm=norm, depth=10,
                   hidden=128)
        rs = [r for r in rs if r["args"]["tag"].startswith("seed-")]
        if not rs:
            continue
        accs = [r["test_accuracy"] * 100 for r in rs]
        tl = [r["train_loss"] for r in rs]
        m, s = agg(accs)
        L(f"{label} & {m:.2f} & {s:.2f} & {np.mean(tl):.4f} \\\\")
    L(r"\bottomrule")
    L(r"\end{tabular}")
    L(r"\end{table}")
    (Path(out_dir) / "tab_seeds.tex").write_text("\n".join(lines),
                                                 encoding="utf-8")


# ============================================================
# 图
# ============================================================

def fig_activation(recs, out_dir):
    rows = []
    for a in ACTS:
        r = one(recs, dataset="mnist", activation=a, depth=10, hidden=128,
                seed=42, tag="mnist-L10")
        if r:
            rows.append((a, r))
    rows.sort(key=lambda z: z[1]["test_accuracy"])
    fig, ax = plt.subplots(figsize=(6.4, 3.1))
    y = np.arange(len(rows))
    accs = [r["test_accuracy"] * 100 for _, r in rows]
    colors = ["#c0392b" if a.startswith("poly") else
              ("#e08214" if a in ("swiglu", "geglu", "reglu", "bilinear",
                                  "seglu", "mishglu", "tanhglu")
               else "#4c72b0") for a, _ in rows]
    ax.barh(y, accs, color=colors, height=0.72)
    ax.set_yticks(y)
    ax.set_yticklabels([NICE.get(a, a) for a, _ in rows])
    ax.set_xlabel("Test accuracy (%)")
    ax.set_xlim(0, 105)
    for yi, v in zip(y, accs):
        ax.text(v + 1, yi, f"{v:.1f}", va="center", fontsize=7)
    fig.savefig(Path(out_dir) / "fig_activation.pdf")
    plt.close(fig)
    return rows


def fig_norm(recs, out_dir):
    combos = [("none", "post", "None"), ("batch", "post", "BN"),
              ("layer", "post", "LN"), ("rms", "post", "RMS"),
              ("layer", "pre", "LN (pre)"), ("rms", "pre", "RMS (pre)"),
              ("l2", "post", r"$\ell_2$ proj.")]
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    w = 0.38
    x = np.arange(len(combos))
    for i, (act, tag, c) in enumerate([("relu", "mnist-L10-relu", "#4c72b0"),
                                       ("poly", "mnist-L10-poly-all",
                                        "#c0392b")]):
        vals, errs = [], []
        for norm, pos, _ in combos:
            rs = index(recs, dataset="mnist", activation=act, norm=norm,
                       norm_pos=pos, depth=10, hidden=128, seed=42, tag=tag)
            vals.append(rs[0]["test_accuracy"] * 100 if rs else np.nan)
            errs.append(0)
        ax.bar(x + (i - 0.5) * w, vals, w, color=c, label=NICE.get(act, act))
        for xi, v in zip(x + (i - 0.5) * w, vals):
            if not math.isnan(v):
                ax.text(xi, v + 0.6, f"{v:.1f}", ha="center", fontsize=6.5,
                        rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels([c[2] for c in combos])
    ax.set_ylabel("Test accuracy (%)")
    ax.set_ylim(80, 100)
    ax.legend(loc="lower right")
    fig.savefig(Path(out_dir) / "fig_norm.pdf")
    plt.close(fig)


def fig_degree_heatmap(recs, out_dir):
    degs, lams = [2, 3, 4, 5], [0.001, 0.01, 0.05, 0.1]
    M = np.full((len(degs), len(lams)), np.nan)
    for i, d in enumerate(degs):
        for j, lm in enumerate(lams):
            r = one(recs, dataset="mnist", activation="poly", degree=d, lam=lm,
                    norm="l2", depth=10, hidden=128, seed=42)
            if r:
                M[i, j] = r["test_accuracy"] * 100
    fig, ax = plt.subplots(figsize=(4.6, 2.6))
    im = ax.imshow(M, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(lams)))
    ax.set_xticklabels([f"{l:g}" for l in lams])
    ax.set_yticks(range(len(degs)))
    ax.set_yticklabels([f"$d={d}$" for d in degs])
    ax.set_xlabel(r"$\lambda$")
    for i in range(len(degs)):
        for j in range(len(lams)):
            if not math.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.1f}", ha="center", va="center",
                        color="w", fontsize=8)
    fig.colorbar(im, ax=ax, label="Test accuracy (%)")
    ax.grid(False)
    fig.savefig(Path(out_dir) / "fig_degree.pdf")
    plt.close(fig)


def fig_depth(recs, out_dir):
    depths = [2, 3, 5, 10, 15, 20, 30, 50]
    cfgs = [("relu", "none", "relu", "ReLU", "#4c72b0", "-o"),
            ("relu", "layer", "relu-ln", "ReLU+LN", "#4c72b0", "--s"),
            ("gelu", "none", "gelu", "GELU", "#55a868", "-^"),
            ("poly", "none", "poly", "Poly", "#c0392b", "-D"),
            ("poly", "l2", "polyproj", r"Poly+$\ell_2$", "#c0392b", "-o"),
            ("poly", "layer", "poly-ln", "Poly+LN", "#c0392b", "--s"),
            ("poly", "batch", "poly-bn", "Poly+BN", "#c0392b", ":v")]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    for act, norm, tag, label, c, ls in cfgs:
        xs, ys = [], []
        for d in depths:
            r = one(recs, dataset="mnist", activation=act, norm=norm, depth=d,
                    hidden=128, seed=42, tag=f"mnist-L{d}-{tag}")
            if r:
                xs.append(d)
                ys.append(r["test_accuracy"] * 100)
        if xs:
            ax.plot(xs, ys, ls, color=c, label=label, ms=4, lw=1.2)
    ax.set_xlabel("Number of linear layers $L$")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_xscale("log")
    ax.set_xticks(depths)
    ax.set_xticklabels([str(d) for d in depths])
    ax.set_ylim(10, 100)
    ax.legend(ncol=2)
    fig.savefig(Path(out_dir) / "fig_depth.pdf")
    plt.close(fig)


def fig_grad_profile(recs, out_dir):
    """梯度范数逐层剖面（训练后），对比深/浅与归一化。"""
    pick = [("relu", "none", 20, "relu-L20", "ReLU, $L{=}20$", "#4c72b0", "-o"),
            ("relu", "layer", 20, "relu-ln", "ReLU+LN, $L{=}20$", "#4c72b0", "--s"),
            ("poly", "none", 20, "poly", "Poly, $L{=}20$", "#c0392b", "-D"),
            ("poly", "l2", 20, "polyproj", r"Poly+$\ell_2$, $L{=}20$", "#c0392b", "-o"),
            ("poly", "layer", 20, "poly-ln", "Poly+LN, $L{=}20$", "#c0392b", "--s"),
            ("poly", "batch", 20, "poly-bn", "Poly+BN, $L{=}20$", "#c0392b", ":v"),
            ("poly", "l2", 50, "polyproj", r"Poly+$\ell_2$, $L{=}50$", "#8e44ad", "-o"),
            ("relu", "layer", 50, "relu-ln", "ReLU+LN, $L{=}50$", "#2c3e50", "--s")]
    fig, ax = plt.subplots(figsize=(6.4, 3.1))
    for act, norm, d, tag, label, c, ls in pick:
        r = one(recs, dataset="mnist", activation=act, norm=norm, depth=d,
                hidden=128, seed=42, tag=f"mnist-L{d}-{tag}")
        if not r or not r.get("grad_norm_end"):
            continue
        g = np.array(r["grad_norm_end"], dtype=float)
        g = np.maximum(g, 1e-14)
        xs = np.arange(len(g))
        ax.semilogy(xs, g, ls, color=c, label=label, ms=3, lw=1.1)
    ax.set_xlabel("Linear layer index (input $\\to$ output)")
    ax.set_ylabel(r"$\|\nabla W_\ell\|_F$")
    ax.legend(ncol=2)
    fig.savefig(Path(out_dir) / "fig_grad_profile.pdf")
    plt.close(fig)


def fig_act_norm(recs, out_dir):
    """逐层激活范数：射影归一化把它钉在 1。"""
    pick = [("poly", "none", 10, "poly", "Poly, norm=none", "#c0392b", "-D"),
            ("poly", "l2", 10, "polyproj", r"Poly+$\ell_2$", "#c0392b", "-o"),
            ("poly", "batch", 10, "poly-bn", "Poly+BN", "#c0392b", ":v"),
            ("poly", "layer", 10, "poly-ln", "Poly+LN", "#c0392b", "--s"),
            ("relu", "none", 10, "relu", "ReLU", "#4c72b0", "-o"),
            ("relu", "layer", 10, "relu-ln", "ReLU+LN", "#4c72b0", "--s")]
    fig, ax = plt.subplots(figsize=(6.4, 3.0))
    for act, norm, d, tag, label, c, ls in pick:
        r = one(recs, dataset="mnist", activation=act, norm=norm, depth=d,
                hidden=128, seed=42, tag=f"mnist-L{d}-{tag}")
        if not r or not r.get("act_norm_end"):
            continue
        v = np.array(r["act_norm_end"], dtype=float)
        ax.semilogy(np.arange(len(v)), np.maximum(v, 1e-12), ls, color=c,
                    label=label, ms=4, lw=1.2)
    ax.axhline(1.0, color="k", ls=":", lw=0.8)
    ax.set_xlabel("Hidden layer index")
    ax.set_ylabel(r"Mean $\|\mathbf{h}^{(\ell)}\|_2$")
    ax.legend(ncol=2)
    fig.savefig(Path(out_dir) / "fig_act_norm.pdf")
    plt.close(fig)


def fig_synth(recs, out_dir):
    """自构数据的激活对比：分组柱状 / 热力图。"""
    names = [("rings", "Rings"), ("spiral", "Spiral"), ("blobs", "Blobs"),
             ("xor", "XOR"), ("checker", "Checker"), ("cubes", "3-bit XOR"),
             ("parity", "Parity"), ("gauss_highdim", "High-dim")]
    acts = ["identity", "relu", "gelu", "silu", "tanh", "poly",
            "swiglu", "geglu"]
    M = np.full((len(acts), len(names)), np.nan)
    for j, (name, _) in enumerate(names):
        for i, a in enumerate(acts):
            r = one(recs, dataset="synth_train", synth_name=name,
                    activation=a, depth=3, hidden=64, seed=0)
            if r:
                M[i, j] = r["test_accuracy"] * 100
    fig, ax = plt.subplots(figsize=(6.6, 3.4))
    im = ax.imshow(M, cmap="RdYlGn", vmin=50, vmax=100, aspect="auto")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels([l for _, l in names], rotation=35, ha="right")
    ax.set_yticks(range(len(acts)))
    ax.set_yticklabels([r"Poly+$\ell_2$" if a == "poly" else NICE.get(a, a)
                        for a in acts])
    ax.grid(False)
    for i in range(len(acts)):
        for j in range(len(names)):
            if not math.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.0f}", ha="center", va="center",
                        fontsize=6.5,
                        color="k" if 55 < M[i, j] < 92 else "w")
    fig.colorbar(im, ax=ax, label="Test accuracy (%)")
    fig.savefig(Path(out_dir) / "fig_synth.pdf")
    plt.close(fig)


def fig_synth_depth(recs, out_dir):
    depths = [3, 5, 10]
    cfgs = [("relu", "none", "ReLU", "#4c72b0", "-o"),
            ("poly", "l2", r"Poly+$\ell_2$", "#c0392b", "-o"),
            ("poly", "none", "Poly", "#c0392b", "-D"),
            ("poly", "batch", "Poly+BN", "#c0392b", ":v"),
            ("poly", "layer", "Poly+LN", "#c0392b", "--s"),
            ("swiglu", "none", "SwiGLU", "#e08214", "-^")]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.7))
    for ax, (sname, slabel) in zip(axes, [("parity", "10-bit Parity"),
                                          ("rings", "Rings")]):
        for act, norm, label, c, ls in cfgs:
            xs, ys = [], []
            for d in depths:
                r = one(recs, dataset="synth_train", synth_name=sname,
                        activation=act, norm=norm, depth=d, hidden=64, seed=0)
                if r:
                    xs.append(d)
                    ys.append(r["test_accuracy"] * 100)
            if xs and len(xs) > 1:
                ax.plot(xs, ys, ls, color=c, label=label, ms=4, lw=1.2)
        ax.set_title(slabel)
        ax.set_xlabel("Linear layers $L$")
        ax.set_xticks(depths)
        ax.set_ylim(30, 102)
    axes[0].set_ylabel("Test accuracy (%)")
    axes[1].legend(ncol=2, fontsize=6)
    fig.savefig(Path(out_dir) / "fig_synth_depth.pdf")
    plt.close(fig)


def fig_seeds(recs, out_dir):
    cfgs = [("relu", "none", "ReLU", "#4c72b0"),
            ("gelu", "none", "GELU", "#55a868"),
            ("poly", "l2", r"Poly+$\ell_2$", "#c0392b"),
            ("poly", "batch", "Poly+BN", "#e67e22"),
            ("poly", "layer", "Poly+LN", "#8e44ad")]
    fig, ax = plt.subplots(figsize=(4.6, 2.7))
    labels, means, stds = [], [], []
    for act, norm, label, c in cfgs:
        rs = [r for r in index(recs, dataset="mnist", activation=act,
                               norm=norm, depth=10, hidden=128)
              if r["args"]["tag"].startswith("seed-")]
        if not rs:
            continue
        m, s = agg([r["test_accuracy"] * 100 for r in rs])
        labels.append(label)
        means.append(m)
        stds.append(s)
    x = np.arange(len(labels))
    ax.bar(x, means, yerr=stds, capsize=3,
           color=[c for _, _, _, c in cfgs[:len(labels)]])
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_ylim(88, 98)
    for xi, (m, s) in zip(x, zip(means, stds)):
        ax.text(xi, m + s + 0.15, f"{m:.2f}", ha="center", fontsize=7)
    fig.savefig(Path(out_dir) / "fig_seeds.pdf")
    plt.close(fig)


# ============================================================
# 主流程
# ============================================================

def table_glu_norm(recs, out_dir):
    """表：门控单元 x 归一化（MNIST 10 层）。"""
    lines = []
    L = lines.append
    L(r"\begin{table}[t]")
    L(r"\centering")
    L(r"\caption{Interaction between the gated units (GLU family) and the "
      r"normalization on MNIST at depth 10. Gated variants use hidden width "
      r"82 so that the parameter count matches the 128-wide elementwise "
      r"networks. Without normalization every gated variant collapses "
      r"numerically; adding normalization restores it.}")
    L(r"\label{tab:glunorm}")
    L(r"\small")
    L(r"\begin{tabular}{lrrrrr}")
    L(r"\toprule")
    L(r"Activation & None & $\ell_2$ & Batch & Layer & RMS \\")
    L(r"\midrule")
    acts = ["swiglu", "geglu", "polyglu", "relu", "gelu", "silu"]
    norms = ["none", "l2", "batch", "layer", "rms"]
    for a in acts:
        cells = []
        for n in norms:
            r = one(recs, dataset="mnist", activation=a, norm=n, depth=10,
                    hidden=128, seed=42)
            cells.append(fmt(r["test_accuracy"] * 100, 2) if r else "--")
        L(f"{NICE.get(a, a)} & " + " & ".join(cells) + r" \\")
    L(r"\bottomrule")
    L(r"\end{tabular}")
    L(r"\end{table}")
    (Path(out_dir) / "tab_glu_norm.tex").write_text("\n".join(lines),
                                                    encoding="utf-8")


def table_learnpoly(recs, out_dir):
    """表：可学习多项式激活（Legendre / Chebyshev / B-spline）。

    每个可学习激活在 none/l2/batch 三种归一化中取最佳配置，并显式列出所用
    归一化器：离开归一化时 Legendre/Chebyshev 会塌陷到随机水平，这一点必须
    在表中可见，否则会得出"可学习多项式无用"的错误结论。
    """
    TB = chr(92) * 2  # 行分隔符

    def best(a):
        cands = index(recs, dataset="mnist", activation=a, depth=10, hidden=128,
                      seed=42)
        cands = [r for r in cands if r.get("test_accuracy") is not None]
        if not cands:
            return None
        return max(cands, key=lambda r: r["test_accuracy"])

    rows = []
    for a in ["legendre", "chebyshev", "bspline"]:
        r = best(a)
        if r:
            rows.append((NICE.get(a, a), r["args"].get("norm"), r))
    # 固定多项式基准：必须钉死为论文的规范配置（degree=3, lam=0.01），
    # 否则 one() 会跨组命中 degree_mnist 里的其它 lam/degree，与表 tab:activation
    # 的同一配置给出不同数字，论文自相矛盾。
    for a, norm in [("poly_proj", "l2"), ("poly", "none")]:
        r = one(recs, dataset="mnist", activation=a, depth=10, hidden=128,
                seed=42, degree=3, lam=0.01, norm=norm)
        if r:
            rows.append((NICE.get(a, a), norm, r))

    lines = []
    L = lines.append
    L(r"\begin{table}[t]")
    L(r"\centering")
    L(r"\caption{Learnable polynomial activations versus the fixed polynomial "
      r"baselines on MNIST at depth 10, each reported at the normaliser that "
      r"serves it best. Without normalisation both Legendre and Chebyshev "
      r"collapse to chance (11.35\%); the B-spline and Chebyshev variants "
      r"with a normaliser are competitive with the modern elementwise "
      r"activations. Single seed.}")
    L(r"\label{tab:learnpoly}")
    L(r"\small")
    L(r"\begin{tabular}{llrrrr}")
    L(r"\toprule")
    L("Activation & Norm. & Params & Train acc.\\ (\\%) & "
      "Test acc.\\ (\\%) & s/epoch " + TB)
    L(r"\midrule")
    for name, norm, r in sorted(rows, key=lambda z: -z[2]["test_accuracy"]):
        L(f"{name} & {norm} & {r['n_params']:,} & "
          f"{r['train_accuracy'] * 100:.2f} & "
          f"{r['test_accuracy'] * 100:.2f} & "
          f"{r['seconds_per_epoch']:.1f} " + TB)
    L(r"\bottomrule")
    L(r"\end{tabular}")
    L(r"\end{table}")
    (Path(out_dir) / "tab_learnpoly.tex").write_text("\n".join(lines),
                                                     encoding="utf-8")
def table_width(recs, out_dir):
    """表：宽度对齐消融。

    注意：harness 里 activation=poly + norm=l2 与 activation=poly_proj +
    norm=none 是同一件事（poly_proj 的定义就是 poly 后接射影 l2 归一化），
    因此 width 组里这两行的数字逐格相同。表中合并为一行、如实注明等价，
    而不是假装它们是两个不同配置（旧版把它们列成两行并让 one() 取到
    degree_mnist 的记录，产出了宽度 128 下的幻值 97.36）。
    """
    lines = []
    L = lines.append
    L(r"\begin{table}[t]")
    L(r"\centering")
    L(r"\caption{Effect of the hidden width on the test accuracy (MNIST, "
      r"depth 10). The polynomial row is the projective polynomial "
      r"($z+\lambda z^{3}$ followed by the $\ell_2$ projection); in this "
      r"harness that is identical to \texttt{poly\_proj}.}")
    L(r"\label{tab:width}")
    L(r"\small")
    L(r"\begin{tabular}{lrrr}")
    L(r"\toprule")
    L(r"Activation & Width 82 & Width 128 & Width 192 \\")
    L(r"\midrule")
    for a, norm in [("relu", "none"), ("gelu", "none"),
                    ("poly_proj", "none")]:
        cells = []
        for w in [82, 128, 192]:
            r = one(recs, out=Path(out_dir).name, dataset="mnist", activation=a,
                    norm=norm, depth=10, hidden=w, seed=42)
            cells.append(fmt(r["test_accuracy"] * 100, 2) if r else "--")
        L(f"{NICE.get(a, a)} & " + " & ".join(cells) + r" \\")
    L(r"\bottomrule")
    L(r"\end{tabular}")
    L(r"\end{table}")
    (Path(out_dir) / "tab_width.tex").write_text("\n".join(lines),
                                                 encoding="utf-8")


def fig_poly_degree(recs, out_dir):
    """图：可学习多项式阶数 vs 测试准确率。"""
    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    Ks, accs = [], []
    for K in [3, 6, 10, 16]:
        r = one(recs, dataset="mnist", activation="legendre", depth=10,
                hidden=128, seed=42, lpoly_degree=K)
        if r:
            Ks.append(K)
            accs.append(r["test_accuracy"] * 100)
    if Ks:
        ax.plot(Ks, accs, "o-", color="#1f77b4", label="Legendre")
    gs, ga = [], []
    for G_ in [4, 8, 16]:
        r = one(recs, dataset="mnist", activation="bspline", depth=10,
                hidden=128, seed=42, bspline_grid=G_)
        if r:
            gs.append(G_)
            ga.append(r["test_accuracy"] * 100)
    if gs:
        ax.plot(gs, ga, "s--", color="#d62728", label="B-spline")
    ax.set_xlabel("Basis degree $K$ / grid size $G$")
    ax.set_ylabel("Test accuracy (%)")
    ax.legend()
    fig.savefig(Path(out_dir) / "fig_learnpoly.pdf")
    plt.close(fig)


def fig_glu_norm(recs, out_dir):
    """图：门控/逐元素激活在不同归一化下的测试准确率（分组柱状图）。"""
    acts = ["swiglu", "geglu", "polyglu", "relu", "gelu", "silu"]
    norms = ["none", "l2", "batch", "layer", "rms"]
    fig, ax = plt.subplots(figsize=(6.2, 3.0))
    w = 0.15
    xs = np.arange(len(acts))
    colors = ["#7f7f7f", "#2ca02c", "#1f77b4", "#ff7f0e", "#9467bd"]
    for j, n in enumerate(norms):
        vals = []
        for a in acts:
            r = one(recs, dataset="mnist", activation=a, norm=n, depth=10,
                    hidden=128, seed=42)
            vals.append(r["test_accuracy"] * 100 if r else np.nan)
        ax.bar(xs + (j - 2) * w, vals, w, label=NICE.get(n, n),
               color=colors[j])
    ax.set_xticks(xs)
    ax.set_xticklabels([NICE.get(a, a) for a in acts], fontsize=7)
    ax.set_ylabel("Test accuracy (%)")
    ax.set_ylim(0, 105)
    ax.legend(ncol=3, fontsize=6.5)
    fig.savefig(Path(out_dir) / "fig_glu_norm.pdf")
    plt.close(fig)


def fig_width(recs, out_dir):
    """图：宽度对齐。"""
    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    for a, norm, c, mk, lb in [("relu", "none", "#1f77b4", "o", "ReLU"),
                               ("gelu", "none", "#ff7f0e", "s", "GELU"),
                               ("poly_proj", "none", "#2ca02c", "^",
                                r"Poly+$\ell_2$"),
                               ("poly", "l2", "#d62728", "D",
                                r"Poly, $\ell_2$ norm")]:
        ws, accs = [], []
        for w in [82, 128, 192]:
            r = one(recs, dataset="mnist", activation=a, norm=norm, depth=10,
                    hidden=w, seed=42)
            if r:
                ws.append(w)
                accs.append(r["test_accuracy"] * 100)
        if ws:
            ax.plot(ws, accs, mk + "-", color=c, label=lb)
    ax.set_xlabel("Hidden width")
    ax.set_ylabel("Test accuracy (%)")
    ax.legend(fontsize=6.5)
    fig.savefig(Path(out_dir) / "fig_width.pdf")
    plt.close(fig)


def write_macros(recs, out_dir):
    """生成 paper/macros.tex：正文引用的所有数字都来自这里，避免手写数字。"""
    M = {}

    def s(name, val):
        M[name] = val

    # 参考网络（复现原脚本）
    r = one(recs, dataset="mnist", activation="poly_proj", norm="none",
            depth=10, hidden=128, seed=42)
    if r:
        s("refTestAcc", f"{r['test_accuracy']*100:.2f}")
        s("refTrainAcc", f"{r['train_accuracy']*100:.2f}")
        s("refParams", f"{r['n_params']:,}")
        s("refEpochSec", f"{r['seconds_per_epoch']:.1f}")
        s("refTotalSec", f"{r['seconds']:.0f}")

    # 激活函数组
    act_rows = []
    for a in ACTS:
        rr = one(recs, dataset="mnist", activation=a, depth=10, hidden=128,
                 seed=42)
        if rr:
            act_rows.append((a, rr))
    if act_rows:
        best = max(act_rows, key=lambda z: z[1]["test_accuracy"])
        worst = min(act_rows, key=lambda z: z[1]["test_accuracy"])
        s("bestActName", NICE.get(best[0], best[0]).replace("$", ""))
        s("bestActAcc", f"{best[1]['test_accuracy']*100:.2f}")
        s("worstActName", NICE.get(worst[0], worst[0]).replace("$", ""))
        s("worstActAcc", f"{worst[1]['test_accuracy']*100:.2f}")
        s("nActCfgs", str(len(act_rows)))
        # 非饱和激活的均值
        good = [rr for a, rr in act_rows
                if rr["test_accuracy"] > 0.5 and a not in ("identity", "sigmoid")]
        if good:
            s("meanGoodAct",
              f"{np.mean([rr['test_accuracy'] for rr in good])*100:.2f}")

    # 门控崩溃
    glu_bad, glu_ok = [], []
    for a in ["swiglu", "geglu", "reglu", "bilinear", "seglu", "mishglu",
              "tanhglu", "polyglu"]:
        rr = one(recs, dataset="mnist", activation=a, norm="none", depth=10,
                 seed=42)
        if rr:
            glu_bad.append(rr["test_accuracy"])
        rl = one(recs, dataset="mnist", activation=a, norm="layer", depth=10,
                 seed=42)
        if rl:
            glu_ok.append(rl["test_accuracy"])
    if glu_bad:
        s("gluNoneMean", f"{np.mean(glu_bad)*100:.2f}")
        s("gluNoneMax", f"{max(glu_bad)*100:.2f}")
        s("nGluNone", str(len(glu_bad)))
    if glu_ok:
        s("gluLayerMean", f"{np.mean(glu_ok)*100:.2f}")
    # polyglu 的对照
    for nm, lbl in [("swiglu", "swiglu"), ("geglu", "geglu")]:
        for norm in ["none", "l2", "layer", "batch", "rms"]:
            rr = one(recs, dataset="mnist", activation=nm, norm=norm,
                     depth=10, seed=42)
            if rr:
                s(f"{lbl}{'None' if norm=='none' else norm.capitalize()}Acc",
                  f"{rr['test_accuracy']*100:.2f}")

    # 归一化组：poly 与 relu 在各归一化下
    for act in ["poly", "relu"]:
        for norm in ["none", "l2", "batch", "layer", "rms"]:
            rr = one(recs, dataset="mnist", activation=act, norm=norm,
                     depth=10, hidden=128, seed=42)
            if rr:
                s(f"{act}{norm.capitalize()}Acc",
                  f"{rr['test_accuracy']*100:.2f}")

    # 多项式阶数
    for deg in [2, 3, 4, 5]:
        rr = one(recs, dataset="mnist", activation="poly", degree=deg,
                 lam=0.01, norm="l2", depth=10, seed=42)
        if rr:
            s(f"deg{deg}Acc", f"{rr['test_accuracy']*100:.2f}")

    # 深度
    for d in [2, 3, 5, 10, 15, 20]:
        for act, norm, lbl in [("relu", "none", "relu"),
                               ("poly", "l2", "polyproj"),
                               ("poly", "batch", "polybn"),
                               ("poly", "layer", "polyln"),
                               ("identity", "none", "identity")]:
            rr = one(recs, dataset="mnist", activation=act, norm=norm,
                     depth=d, seed=42)
            if rr:
                s(f"L{d}{lbl.capitalize()}Acc", f"{rr['test_accuracy']*100:.2f}")

    # 多 seed 均值/标准差
    for lbl, q in [("relu", dict(activation="relu", norm="none")),
                   ("polyproj", dict(activation="poly", norm="l2")),
                   ("polyln", dict(activation="poly", norm="layer")),
                   ("pybn", dict(activation="poly", norm="batch"))]:
        vals = [rr["test_accuracy"] for rr in
                index(recs, dataset="mnist", depth=10, hidden=128, **q)]
        if len(vals) > 1:
            s(f"seeds{lbl.capitalize()}Mean", f"{np.mean(vals)*100:.2f}")
            s(f"seeds{lbl.capitalize()}Std", f"{np.std(vals, ddof=1)*100:.2f}")
            s(f"seeds{lbl.capitalize()}N", str(len(vals)))

    # 隐藏层激活范数比（梯度剖面）：poly vs relu
    for lbl, q in [("poly", dict(activation="poly", norm="l2")),
                   ("relu", dict(activation="relu", norm="none"))]:
        rr = one(recs, dataset="mnist", depth=10, **q)
        if rr and rr.get("act_norm_init"):
            an = rr["act_norm_init"]
            s(f"actNorm{lbl.capitalize()}First", f"{an[0]:.2f}")
            s(f"actNorm{lbl.capitalize()}Last", f"{an[-1]:.2f}")
            s(f"actNorm{lbl.capitalize()}Ratio",
              f"{an[-1]/max(an[0],1e-9):.3f}")

    lines = ["% 由 experiments/analyze.py 自动生成——请勿手改。",
             "% 所有正文数字均引用此文件的宏。"]
    for k in sorted(M):
        lines.append(r"\newcommand{\%s}{%s}" % (k, M[k]))
    (Path(out_dir) / "macros.tex").write_text("\n".join(lines) + "\n",
                                              encoding="utf-8")
    return M


def table_ref(recs, out_dir):
    """表：参考实现的逐位复现。

    这是论文可信度的基石 —— 全部数字来自 harness 自身重跑的结果目录
    （ref_orig = 20 轮、ref_fidelity = 2 轮），并与随网络一起发布的
    results.json 对照。epoch 1 的 train/test 与原始日志逐位相同，说明
    复现不是"数量级接近"而是"打印精度内一致"；epoch 2 起参考组与
    保真组因余弦退火的 T_max 取 args.epochs 而分叉，已在正文说明。
    """
    lines = []
    L = lines.append
    L(r"\begin{table}[t]")
    L(r"\centering")
    L(r"\caption{Reproducing the reference network (depth-ten projective "
      r"polynomial, no separate normalisation) with the unified harness. "
      r"The published \texttt{results.json} reports $\refTestAcc\%$ test "
      r"accuracy after $\refEpochs$ epochs. Epoch-one train/test values "
      r"agree to the precision printed in the original logs; epoch two "
      r"onward the trajectory depends on the annealing budget "
      r"($T_{\max}=\texttt{epochs}$), which differs between the 20-epoch "
      r"reference run and the 2-epoch fidelity check.}")
    L(r"\label{tab:ref}")
    L(r"\small")
    L(r"\begin{tabular}{lrr}")
    L(r"\toprule")
    L(r"Run & Train (\%) & Test (\%) \\")
    L(r"\midrule")
    L(r"Reference, epoch 1 & $\refEoneTrain$ & $\refEoneTest$ \\")
    L(r"Reference, epoch 2 & $\refEtwoTrain$ & $\refEtwoTest$ \\")
    L(r"Reference, final & $\refTrainAcc$ & $\refTestAcc$ \\")
    L(r"\midrule")
    L(r"Harness (2\,ep), epoch 1 & $\fidEOneTrain$ & $\fidEOneTest$ \\")
    L(r"Harness (2\,ep), epoch 2 & $\fidETwoTrain$ & $\fidETwoTest$ \\")
    L(r"\bottomrule")
    L(r"\end{tabular}")
    L(r"\end{table}")
    (Path(out_dir) / "tab_ref.tex").write_text("\n".join(lines),
                                               encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", default="experiments")
    ap.add_argument("--table-dir", default="paper/tables")
    ap.add_argument("--fig-dir", default="paper/figures")
    args = ap.parse_args()

    Path(args.table_dir).mkdir(parents=True, exist_ok=True)
    Path(args.fig_dir).mkdir(parents=True, exist_ok=True)

    recs = load_all(args.exp_dir)
    print(f"载入 {len(recs)} 条实验记录")
    by_group = defaultdict(int)
    for r in recs:
        by_group[r["config_id"].split("-")[0]] += 1
    for k, v in sorted(by_group.items()):
        print(f"  {k:24s} {v}")

    table_activation(recs, args.table_dir)
    table_norm(recs, args.table_dir)
    grid = table_degree(recs, args.table_dir)
    table_depth(recs, args.table_dir)
    table_synth(recs, args.table_dir)
    table_synth_norm(recs, args.table_dir)
    table_seeds(recs, args.table_dir)
    table_glu_norm(recs, args.table_dir)
    table_learnpoly(recs, args.table_dir)
    table_width(recs, args.table_dir)
    table_ref(recs, args.table_dir)
    fig_activation(recs, args.fig_dir)
    fig_norm(recs, args.fig_dir)
    fig_degree_heatmap(recs, args.fig_dir)
    fig_depth(recs, args.fig_dir)
    fig_grad_profile(recs, args.fig_dir)
    fig_act_norm(recs, args.fig_dir)
    fig_synth(recs, args.fig_dir)
    fig_synth_depth(recs, args.fig_dir)
    fig_seeds(recs, args.fig_dir)
    fig_poly_degree(recs, args.fig_dir)
    fig_glu_norm(recs, args.fig_dir)
    fig_width(recs, args.fig_dir)
    # 注意：paper/macros.tex 由 experiments/make_macros.py 单独生成，
    # 这里不再覆写，避免两份生成器互相覆盖。
    print(f"表格 -> {args.table_dir}")
    print(f"图   -> {args.fig_dir}")


if __name__ == "__main__":
    main()
