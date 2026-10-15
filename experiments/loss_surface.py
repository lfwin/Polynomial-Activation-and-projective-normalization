#!/usr/bin/env python3
"""
小块损失曲面的绘制数据：$d = 1, 2, 3$ 个训练参数的简单模型，
标注关键点（临界点 / 全局极小），并把每个关键点的损失值写进 JSON。

模型（未截断 $\varphi(z) = z + \lambda z^{3}$，$\lambda = 0.01$，
square loss，与 experiments/algebraic_table.sage 同一设定）：

  d=1  : $m=1$, 单样本，令 $w_1 = v_1 = t$（两权重共享一个自由度）
         $\hat y = t\,\varphi(t x)$，$L(t) = (\hat y - y)^2$
  d=2  : $m=1$，两样本，参数 $(w_1, v_1)$，$\hat y_a = v_1 \varphi(w_1 x_a)$
  d=3  : R1 残差、深度 2、单宽度，单样本，参数 $(w_1, w_2, v)$
         $h^1 = \varphi(w_1 x)$, $h^2 = h^1 + \varphi(w_2 h^1)$, $\hat y = v h^2$

关键点来源分两类，分清：
  * 代数临界点：由 experiments/algebraic_table.sage 的 Groebner 计算给出
    （上面的 $d=1,2$ 情形在实轴上只有原点一个）；这里用 autograd 复核
    $\lVert \nabla L \rVert \approx 0$。
  * 全局极小：用多起点梯度下降数值求得（不同起点取最好）。

输出：
  experiments/loss_surface.json      关键点坐标 + 损失 + 梯度范数
  paper/figures/fig_loss_surface.pdf 三联图（1D 曲线 / 2D 等高线 / 3D 曲面）

跑法（限 10 核）：
  experiments/with_cpu_cap.sh ../.venv/bin/python experiments/loss_surface.py
"""
import json
import math
import os
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

torch.set_num_threads(min(10, torch.get_num_threads()))

LAM = 0.01
SAMPLES = [(0.5, 1.0), (-1.2, -0.4), (2.0, 0.3)]
ROOT = Path(__file__).resolve().parent
OUT_JSON = ROOT / "loss_surface.json"
OUT_FIG = ROOT.parent / "paper" / "figures" / "fig_loss_surface.pdf"


def phi(z):
    return z + LAM * z ** 3


# --------------------------------------------------------------- 损失函数
def loss_d1(t, x, y):
    """t 为张量；返回逐点 1-维损失。"""
    return (t * phi(t * x) - y) ** 2


def loss_d2_grid(W1, V1, xs, ys):
    """W1,V1 同形状网格 -> 损失网格。"""
    L = torch.zeros_like(W1)
    for x, y in zip(xs, ys):
        h = phi(W1 * x)
        L = L + (V1 * h - y) ** 2
    return L


def loss_d3(W1, W2, V, x, y):
    h1 = phi(W1 * x)
    h2 = h1 + phi(W2 * h1)
    return (V * h2 - y) ** 2


def grad_norm_d1(t, x, y):
    t = torch.tensor(float(t), dtype=torch.float64, requires_grad=True)
    L = loss_d1(t, x, y)
    g, = torch.autograd.grad(L, t)
    return float(L), float(g.abs())


def grad_norm_d2(w1, v1, xs, ys):
    a = torch.tensor([[float(w1)]], dtype=torch.float64, requires_grad=True)
    b = torch.tensor([[float(v1)]], dtype=torch.float64, requires_grad=True)
    L = loss_d2_grid(a, b, xs, ys).sum()
    g = torch.autograd.grad(L, [a, b])
    return float(L), float(torch.sqrt(sum(x.pow(2).sum() for x in g)))


def grad_norm_d3(w1, w2, v, x, y):
    a = torch.tensor(float(w1), dtype=torch.float64, requires_grad=True)
    b = torch.tensor(float(w2), dtype=torch.float64, requires_grad=True)
    c = torch.tensor(float(v), dtype=torch.float64, requires_grad=True)
    L = loss_d3(a, b, c, x, y)
    g = torch.autograd.grad(L, [a, b, c])
    return float(L), float(torch.sqrt(sum(x.pow(2) for x in g)))


# --------------------------------------------------------------- 极小值搜索
def optimise(fn, dim, bounds, n_starts=160, steps=600, seed=0):
    """多起点 Adam，返回 (最好坐标, 最好损失)。fn 接受 (N, dim) 张量。"""
    g = torch.Generator().manual_seed(seed)
    lo = torch.tensor([b[0] for b in bounds], dtype=torch.float64)
    hi = torch.tensor([b[1] for b in bounds], dtype=torch.float64)
    X = lo + (hi - lo) * torch.rand(n_starts, dim, generator=g, dtype=torch.float64)
    X.requires_grad_(True)
    opt = torch.optim.Adam([X], lr=0.05)
    best = (None, math.inf)
    for _ in range(steps):
        opt.zero_grad()
        L = fn(X).sum()
        L.backward()
        opt.step()
        with torch.no_grad():
            vals = fn(X)
            j = int(torch.argmin(vals))
            if float(vals[j]) < best[1]:
                best = (X[j].detach().clone(), float(vals[j]))
    return [float(v) for v in best[0]], best[1]


# --------------------------------------------------------------- 主流程
def main():
    data = {}

    # ---- d = 1 ----
    x1, y1 = SAMPLES[0]
    t_grid = torch.linspace(-1.6, 1.6, 601, dtype=torch.float64)
    L1 = loss_d1(t_grid, x1, y1)
    t_min, L_min_1 = optimise(
        lambda T: loss_d1(T[:, 0], x1, y1), 1, [(-1.6, 1.6)], seed=1)
    # 这条切片上的**全部**实临界点：解析上 dL/dt = t(t^2+200)(t^4+400t^2-800)/80000，
    # 零点为 t = 0 与 t = ±sqrt(20*sqrt(102) - 200) ≈ ±1.4107。列出全部，不要只报一个。
    crit = [0.0]
    for rt in np.roots([1.0, 0.0, 400.0, 0.0, -800.0]):
        if abs(rt.imag) < 1e-12 and abs(rt.real) <= 1.6:
            crit.append(float(rt.real))
    crit.sort()
    pts1 = []
    for c in crit:
        Lc, gc = grad_norm_d1(c, x1, y1)
        pts1.append({"name": "t0" if c == 0.0 else "crit",
                     "coord": [c], "loss": Lc, "gradnorm": gc,
                     "kind": "critical point of the slice (algebraic)"})
    # 极小点与其对应的临界点合并（数值极小应与某个临界点重合）
    tmin_c = min(crit, key=lambda c: abs(c - float(t_min[0])))
    for p in pts1:                      # 把数值极小标记到已列出的临界点上
        if abs(p["coord"][0] - tmin_c) < 1e-9:
            p["is_min"] = True
    data["d1"] = {
        "x": [float(v) for v in t_grid[::8]],
        "loss": [float(v) for v in L1[::8]],
        "critical": crit,
        "linear_min": float(math.sqrt(2.0)),   # lambda = 0 时的极小点 ±sqrt(2)
        "points": pts1,
    }
    print(f"[d=1] 实临界点 t={['%.4f' % c for c in crit]} "
          f"| L at t=0: {grad_norm_d1(0.0, x1, y1)[0]:.6f} "
          f"| min L={L_min_1:.2e} at t={tmin_c:.4f} (lambda=0 时 sqrt2=%.4f)"
          % math.sqrt(2.0))

    # ---- d = 2 ----
    xs2 = [SAMPLES[0][0], SAMPLES[1][0]]
    ys2 = [SAMPLES[0][1], SAMPLES[1][1]]
    ax = torch.linspace(-1.2, 1.2, 241, dtype=torch.float64)
    av = torch.linspace(-8.0, 8.0, 241, dtype=torch.float64)
    W1, V1 = torch.meshgrid(ax, av, indexing="ij")
    L2 = loss_d2_grid(W1, V1, xs2, ys2)
    L0_2, gn_2 = grad_norm_d2(0.0, 0.0, xs2, ys2)
    wmin, L_min_2 = optimise(
        lambda P: loss_d2_grid(P[:, :1], P[:, 1:2], xs2, ys2).reshape(-1), 2,
        [(-1.2, 1.2), (-8.0, 8.0)], seed=2)
    gn_min2 = grad_norm_d2(wmin[0], wmin[1], xs2, ys2)[1]
    # 关键：这个下确界**不是**临界点。w1 -> 0 时 phi(w1 x) -> w1 x，模型退化成过原点的
    # 直线 yhat = (v1 w1) x，最优残差是直线回归的残差。取 x=(x1,x2), y=(y1,y2)：
    #   inf L = ||y||^2 - <x,y>^2/||x||^2,  在 w1->0, v1->inf 处趋近、永不达到。
    # 目标是两条样本 (1/2,1)、(-6/5,-2/5)，符号相反：要插值需要比值 y1/y2 = -5/2，
    # 而模型能取到的比值 |phi(w1 x1)/phi(w1 x2)| 落在 (5/72, 5/12]，即恒 <= 5/12，
    # 离 5/2 差一个数量级 —— 这正是 dim I = 0、degree 5 里四个复临界点没有实对应物的
    # 可视形式：两侧符号无法同时满足。
    xv = [float(SAMPLES[i][0]) for i in range(2)]
    yv = [float(SAMPLES[i][1]) for i in range(2)]
    nx2 = sum(v * v for v in xv)
    yx = sum(xv[i] * yv[i] for i in range(2))
    ny2 = sum(v * v for v in yv)
    inf_exact = ny2 - yx * yx / nx2          # = 100/169
    data["d2"] = {
        "grid_w": [float(v) for v in ax[::4]], "grid_v": [float(v) for v in av[::4]],
        "loss": [[float(L2[i, j]) for j in range(0, 241, 4)] for i in range(0, 241, 4)],
        "points": [
            {"name": "origin", "coord": [0.0, 0.0], "loss": L0_2, "gradnorm": gn_2,
             "kind": "critical (algebraic, only real one)"},
            {"name": "inf", "coord": [0.0, None], "loss": inf_exact,
             "gradnorm": None,
             "kind": "infimum, not attained: $w_1\\to 0$, $v_1\\to\\infty$"},
        ],
    }
    print(f"[d=2] (0,0): L={L0_2:.6f} |g|={gn_2:.2e} | "
          f"inf L={inf_exact:.6f} (w1->0, v1->inf; grid min {L_min_2:.6f} at "
          f"({wmin[0]:.4f},{wmin[1]:.4f}), |g|={gn_min2:.1e})")

    # ---- d = 3 ----
    x3, y3 = SAMPLES[0]
    vmin, L_min_3 = optimise(
        lambda P: loss_d3(P[:, :1], P[:, 1:2], P[:, 2:3], x3, y3).reshape(-1), 3,
        [(-1.2, 1.2), (-3.0, 3.0), (-6.0, 6.0)], seed=3)
    gn_min3 = grad_norm_d3(vmin[0], vmin[1], vmin[2], x3, y3)[1]
    # 面板固定 v 在最优值上，(w1, w2) 两个方向画曲面，这样图上标注的极小点
    # (w1*, w2*) 确实落在该切片上：在 v = v* 处 L(w1*, w2*; v*) = 0。
    r = float(vmin[2])
    n = 49
    g1 = torch.linspace(-1.2, 1.2, n, dtype=torch.float64)
    g2 = torch.linspace(-3.0, 3.0, n, dtype=torch.float64)
    G1, G2 = torch.meshgrid(g1, g2, indexing="ij")
    L3 = loss_d3(G1, G2, torch.full_like(G1, r), x3, y3)
    # (0,0,0) 是该臂唯一的原点临界点：yhat = v h^2 在 v=0 时对 w 的导数整体消失。
    wv = [0.0, 0.0, 0.0]
    L_v0, gn_v0 = grad_norm_d3(*wv, x3, y3)
    data["d3"] = {
        "x": [float(v) for v in g1], "y": [float(v) for v in g2],
        "z_fixed": r,
        "loss": [[float(L3[i, j]) for j in range(n)] for i in range(n)],
        "points": [
            {"name": "origin", "coord": wv, "loss": L_v0, "gradnorm": gn_v0,
             "kind": "critical: $\\lVert\\nabla L\\rVert = 0$ at $w_1=w_2=v=0$"},
            {"name": "min", "coord": vmin, "loss": L_min_3, "gradnorm": gn_min3,
             "kind": "global minimum (numerical), $L = 0$"},
        ],
    }
    print(f"[d=3] (0,0,0): L={L_v0:.6f} |g|={gn_v0:.2e} | "
          f"min=({vmin[0]:.4f},{vmin[1]:.4f},{vmin[2]:.4f}) L={L_min_3:.6f}")

    # ---- 同题对照：两样本上，残差臂能否真的插值，而 plain 臂不能 ----
    # 用同一对样本 (1/2,1),(-6/5,-2/5)，多起点 LBFGS 拟合深度2宽1的残差臂。
    xt = torch.tensor([SAMPLES[0][0], SAMPLES[1][0]], dtype=torch.float64)
    yt = torch.tensor([SAMPLES[0][1], SAMPLES[1][1]], dtype=torch.float64)

    def loss_res2(P):
        w1, w2, v = P[:, :1], P[:, 1:2], P[:, 2:3]
        h1 = phi(w1 * xt)
        h2 = h1 + phi(w2 * h1)
        return ((v * h2 - yt) ** 2).sum(dim=1)

    _, L_res2 = optimise(loss_res2, 3,
                         [(-4.0, 4.0), (-3.0, 3.0), (-6.0, 6.0)], seed=4)

    def loss_plain2(P):
        w1, w2, v = P[:, :1], P[:, 1:2], P[:, 2:3]
        h2 = phi(w2 * phi(w1 * xt))
        return ((v * h2 - yt) ** 2).sum(dim=1)

    _, L_plain2 = optimise(loss_plain2, 3,
                           [(-4.0, 4.0), (-3.0, 3.0), (-6.0, 6.0)], seed=4)
    data["twosample"] = {"residual": L_res2, "plain": L_plain2}
    print(f"[twosample] 残差 L={L_res2:.3e} | plain L={L_plain2:.6f} "
          f"(下确界 {inf_exact:.6f})")

    OUT_JSON.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    print(f"[写出] {OUT_JSON}")

    # ------------------------------------------------------------ 图
    fig = plt.figure(figsize=(6.6, 2.5))
    ax0 = fig.add_subplot(1, 3, 1)
    ax1 = fig.add_subplot(1, 3, 2)
    ax2 = fig.add_subplot(1, 3, 3, projection="3d")

    tt = np.array(data["d1"]["x"])
    ax0.plot(tt, np.array(data["d1"]["loss"]), color="#c0392b", lw=1.4)
    for p in data["d1"]["points"]:
        c = p["coord"][0]
        is_min = p.get("is_min", False)
        ax0.plot([c], [p["loss"]], "o", ms=6, mfc="none", mew=1.5,
                 color="#2a7f3f" if is_min else "#1f4e9c")
        ax0.annotate(f"{p['loss']:.2f}", (c, p["loss"]),
                     textcoords="offset points",
                     xytext=(4, 8) if c >= 0 else (-16, 8), fontsize=6.5)
    ax0.set_xlabel(r"shared weight $t$")
    ax0.set_ylabel(r"square loss $L$")
    ax0.set_title(r"$d=1$: slice $w_1=v_1=t$", fontsize=8)
    ax0.tick_params(labelsize=7)

    W = np.array(data["d2"]["grid_w"])
    V = np.array(data["d2"]["grid_v"])
    Z = np.array(data["d2"]["loss"])          # Z[i,j] -> (w1 index, v1 index)
    im = ax1.pcolormesh(V, W, Z, cmap="viridis", shading="auto",
                        norm=matplotlib.colors.LogNorm(vmin=max(Z.min(), 1e-3),
                                                       vmax=Z.max()))
    for p in data["d2"]["points"]:
        # 下确界在 w1 = 0 的边界上（v1 -> inf），用空心圈标在左边界，不作临界点。
        if p["name"] == "origin":
            ax1.plot(p["coord"][1], p["coord"][0], "*", ms=9, mfc="none",
                     mew=1.3, color="w")
        else:
            # 下确界在边界 w1 = 0 上取到（v1 -> inf），画成虚线并加文字标注。
            ax1.axhline(p["coord"][0], color="#ff5252", lw=1.0, ls=":")
            ax1.text(0.04, 0.90, rf"$\inf\,L={p['loss']:.3f}$",
                     transform=ax1.transAxes, color="#ff5252", fontsize=6)
    ax1.set_xlabel(r"$v_1$")
    ax1.set_ylabel(r"$w_1$")
    ax1.set_title(r"$d=2$: one unit, two samples", fontsize=8)
    ax1.tick_params(labelsize=7)
    fig.colorbar(im, ax=ax1, pad=0.02).ax.tick_params(labelsize=6)

    X3 = np.array(data["d3"]["x"])
    Y3 = np.array(data["d3"]["y"])
    Z3 = np.log10(np.array(data["d3"]["loss"]) + 1e-3)
    Xg, Yg = np.meshgrid(X3, Y3, indexing="ij")
    ax2.plot_surface(Xg, Yg, Z3, cmap="viridis", rstride=1, cstride=1,
                     linewidth=0, antialiased=True, alpha=0.92)
    p = data["d3"]["points"][1]
    ax2.scatter([p["coord"][0]], [p["coord"][1]],
                [math.log10(p["loss"] + 1e-3)], color="#ff5252", s=18, marker="*")
    q = data["d3"]["points"][0]
    ax2.scatter([q["coord"][0]], [q["coord"][1]],
                [math.log10(q["loss"] + 1e-3)], color="w", s=18, marker="o")
    ax2.set_xlabel(rf"$w_1$", fontsize=7)
    ax2.set_ylabel(rf"$w_2$", fontsize=7)
    ax2.set_zlabel(r"$\log_{10} L$", fontsize=6)
    ax2.set_title(rf"$d=3$: residual, $v={r:.3f}$", fontsize=8)
    ax2.tick_params(labelsize=6)

    fig.tight_layout()
    fig.savefig(OUT_FIG)
    plt.close(fig)
    print(f"[写出] {OUT_FIG}")


if __name__ == "__main__":
    main()
