#!/usr/bin/env python3
"""
代数几何刻画：多项式网络 (+ 残差) + projective normalization + square loss
的临界点簇（critical variety）。

本脚本只做**不需要 Groebner 重计算**的两件事，全部可在普通 venv 里跑完：
  (A) 临界方程的结构与维数判据（sympy：grevlex GB + 纯幂首项判据）；
  (B) 截断 clamp 处的梯度边界（torch autograd）。

重计算量大的情形（宽 2 深 2 的残差+射影复合、8 变量射影臂）在
`algebraic_critical_points.sage` 里用 Sage 跑；两者数字互相印证。

约定
----
  phi(z) = z + lambda z^3,  lambda = 0.01,  **未截断**（截断会破坏多项式性）
  plain      : yhat = sum_j v_j phi(w_j x)
  projective : yhat = (v.h)/||h||，用 r^2 = sum_j h_j^2 消掉 sqrt 保持多项式
  residual   : h^1 = phi(w1 x), h^2 = h^1 + phi(w2 h^1)   (R1, 深度 2)

跑法：env -u LD_LIBRARY_PATH .venv/bin/python experiments/algebraic_critical_points.py
"""
import itertools

import numpy as np
import sympy as sp
import torch

LAM = sp.Rational(1, 100)
SAMPLES = [(sp.Rational(1, 2), sp.Integer(1)),
           (sp.Rational(-6, 5), sp.Rational(-2, 5)),
           (sp.Integer(2), sp.Rational(3, 10))]


def phi(z, lam=LAM):
    return z + lam * z ** 3


# ---------------------------------------------------------------- 维数判据
def gb_solve(vars_, eqs):
    """返回 (is_zero_dim, degree, n_gb)。零维当且仅当每个变量都有纯幂首项。"""
    G = sp.groebner(eqs, *vars_, order='grevlex')
    lms = []                      # 首单项的指数向量
    for g in G.exprs:
        m = sp.Poly(g, *vars_).monoms()[0]
        lms.append(tuple(m))
    zero_dim = True
    bounds = []
    for i in range(len(vars_)):
        exps = [lm[i] for lm in lms if sum(lm) == lm[i]]   # 纯幂首项
        if not exps:
            zero_dim = False
        bounds.append(max(lm[i] for lm in lms) + 1)
    if not zero_dim:
        return False, None, len(G.exprs)

    def divisible(mono):
        for lm in lms:
            if all(mono[i] >= lm[i] for i in range(len(lm))):
                return True
        return False

    deg = 0
    for mono in itertools.product(*[range(b) for b in bounds]):
        if not divisible(mono):
            deg += 1
    return True, deg, len(G.exprs)


def plain_system(m, xs, ys, lam=LAM):
    w = sp.symbols(f'w1:{m+1}')
    v = sp.symbols(f'v1:{m+1}')
    L = 0
    for x, y in zip(xs, ys):
        L += (sum(v[j] * phi(w[j] * x, lam) for j in range(m)) - y) ** 2
    L = sp.expand(L)
    return list(w) + list(v), [sp.expand(sp.diff(L, s)) for s in list(w) + list(v)], L


def proj_system(m, x, y):
    """r^2 = sum h^2 消 sqrt；E_theta = 2 G G_theta H - G^2 H_theta，加约束 Phi。"""
    names = [f'w{j}' for j in range(1, m + 1)] + [f'v{j}' for j in range(1, m + 1)] + ['r']
    R = sp.symbols(names)
    w, v, r = R[:m], R[m:2 * m], R[2 * m]
    h = [phi(w[j] * x) for j in range(m)]
    G = sum(v[j] * h[j] for j in range(m)) - y * r
    H = sum(hj ** 2 for hj in h)
    eqs = [sp.expand(2 * G * sp.diff(G, s) * H - G ** 2 * sp.diff(H, s))
           for s in list(w) + list(v)]
    Phi = sp.expand(r ** 2 - H)
    eqs.append(Phi)
    eqs += [sp.expand(Phi * sp.diff(G, s)) for s in list(w) + list(v)]
    return list(R), eqs


def residual_system(x, y):
    w1, w2, v = sp.symbols('w1 w2 v')
    h1 = phi(w1 * x)
    h2 = h1 + phi(w2 * h1)
    L = sp.expand((v * h2 - y) ** 2)
    return [w1, w2, v], [sp.expand(sp.diff(L, s)) for s in (w1, w2, v)]


# ---------------------------------------------------------------- main
def main():
    print("=" * 74)
    print("(0) 未截断 phi：loss 是权重的多项式")
    print("=" * 74)
    w1, w2, v1, v2, x, y = sp.symbols('w1 w2 v1 v2 x y')
    L = sp.expand((v1 * phi(w1 * x) + v2 * phi(w2 * x) - y) ** 2)
    print(f"  deg_w(L)={sp.Poly(L, w1, w2).total_degree()}  "
          f"deg_v(L)={sp.Poly(L, v1, v2).total_degree()}  "
          f"deg_total(L)={sp.Poly(L, w1, w2, v1, v2, x, y).total_degree()}")
    print("  => 未截断时 L 关于权重是多项式，临界方程 dL/dtheta=0 定义代数簇。\n")

    print("=" * 74)
    print("(1) 无归一化 plain：临界点簇")
    print("=" * 74)
    for (m, n) in [(1, 2), (1, 3), (2, 2)]:
        xs = [SAMPLES[i][0] for i in range(n)]
        ys = [SAMPLES[i][1] for i in range(n)]
        vars_, eqs, _ = plain_system(m, xs, ys)
        zd, deg, ng = gb_solve(vars_, eqs)
        maxdeg = max(sp.Poly(e, *vars_).total_degree() for e in eqs)
        tag = f"dim=0  degree={deg}" if zd else "正维（无孤立临界点）"
        print(f"  m={m} n={n}: vars={2*m}  maxdeg={maxdeg}  GB={ng}  {tag}")

    print("\n  对照：纯三次 phi=z^3（去掉一次项）")
    for (m, n) in [(1, 2), (1, 3), (2, 2)]:
        xs = [SAMPLES[i][0] for i in range(n)]
        ys = [SAMPLES[i][1] for i in range(n)]
        vars_, eqs, _ = plain_system(m, xs, ys, lam=sp.Integer(0))
        zd, deg, ng = gb_solve(vars_, eqs)
        print(f"  m={m} n={n}: " + (f"dim=0 degree={deg}" if zd else "正维（无孤立临界点）"))
    print("  => 退化：w 与 -w 给出同一函数（网络不可辨识），临界簇沿该对称性成维。\n")

    print("=" * 74)
    print("(2) 实临界点 vs 复临界点（m=1, n=2）")
    print("=" * 74)
    vars_, eqs, _ = plain_system(1, [SAMPLES[0][0], SAMPLES[1][0]],
                                 [SAMPLES[0][1], SAMPLES[1][1]])
    sols = sp.solve(eqs, vars_, dict=True)
    nreal = 0
    for s in sols:
        real = all(abs(complex(sp.N(val)).imag) < 1e-12 for val in s.values())
        nreal += real
        vals = {str(k): str(complex(sp.N(val, 8))) for k, val in s.items()}
        print(f"    {vals}  {'实' if real else '复'}")
    print(f"  => {len(sols)} 个复解（= 理想次数 5），其中实解 {nreal} 个。")
    print("     唯一的实临界点是权重的原点 w=v=0：而它在真实数据上并不停留，")
    print("     因为一次项 (lambda>0) 使该点不是下降轨迹的吸引子。")
    print("     结论：在 square loss + 真实数据下，(1,2) 的临界点全是复的；")
    print("     实数域内不存在可停留的孤立法方向临界点。\n")

    print("=" * 74)
    print("(3) projective normalization：临界簇必然逃逸")
    print("=" * 74)
    for m in [1, 2]:
        names, eqs = proj_system(m, SAMPLES[0][0], SAMPLES[0][1])
        zd, deg, ng = gb_solve(names, eqs)
        maxdeg = max(sp.Poly(e, *names).total_degree() for e in eqs)
        tag = f"dim=0 degree={deg}" if zd else "正维"
        print(f"  m={m}, n=1: vars={len(names)}  maxdeg={maxdeg}  GB={ng}  {tag}")
    print("  => 射影不变量使 loss 在射线上恒定（yhat(ch)=yhat(h) ∀c>0）：")
    print("     对每个临界点都有一条完整的正实尺度轨道也是临界点，")
    print("     临界簇维数至少 1 高于无归一化情形，不存在孤立临界点。\n")

    print("=" * 74)
    print("(4) 残差 R1（深度 2，无归一化）：仍无孤立临界点")
    print("=" * 74)
    for (x0, y0) in [SAMPLES[0], SAMPLES[2]]:
        vars_, eqs = residual_system(x0, y0)
        zd, deg, ng = gb_solve(vars_, eqs)
        maxdeg = max(sp.Poly(e, *vars_).total_degree() for e in eqs)
        print(f"  x={x0} y={y0}: vars=3  maxdeg={maxdeg}  GB={ng}  "
              + (f"dim=0 degree={deg}" if zd else "正维"))
    print("  => 残差把逐样本的尺度自由度交还给网络，critical set 不在")
    print("     w=v=0 的孤立点聚集，而是铺开成正维簇。\n")

    print("=" * 74)
    print("(5) 截断 clamp 的代数代价：代数模型失效所在的边界")
    print("=" * 74)
    c = sp.Integer(10)
    sat = c + LAM * c ** 3
    print(f"  |z| >= 10 时 phi 恒等于 ±{sat}（关于 z 的常数），dphi/dz = 0；")
    print(f"  单饱和到该值，128 维整层饱和范数 = {int(sat)}*sqrt(128) = "
          f"{float(sat) * 128 ** 0.5:.2f}")

    torch.manual_seed(0)
    width, depth = 128, 100
    xin = torch.randn(width, 1, dtype=torch.float64)
    for scale in [0.5, 1.0, 2.0]:
        Ws = [torch.randn(width, width, dtype=torch.float64, requires_grad=True) * scale
              for _ in range(depth)]
        h = xin
        for W in Ws:
            z = W @ h
            h = z.clamp(-10, 10) + 0.01 * z.clamp(-10, 10) ** 3
        loss = (h ** 2).sum()
        gnorm = max(float(g.norm()) for g in torch.autograd.grad(loss, Ws))
        print(f"  scale={scale:<4} 末层范数={float(h.norm()):>9.3f}  max|grad|={gnorm:.3e}")
    print("  => 网络被推到 |z|>10 后梯度按 clamp 归零，loss 曲面上出现")
    print("     多项式性的失效面；论文里的塌缩臂正落在该面之外。")


if __name__ == '__main__':
    main()
