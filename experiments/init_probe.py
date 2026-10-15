#!/usr/bin/env python3
"""初始化相关的可验证断言（论文「如何初始化」一节的数字来源）。

三个断言，全部在真实前向/反向里测，不靠直觉：

(A) 射影不变性 + 层范数剖面：
    (A1) 投影前的激活整体乘 c，网络输出不变（这正是代数侧「损失沿缩放轨道
         不变」的可测形式）；
    (A2) 初始化时逐层激活范数剖面：射影臂每层恒为 1（确定结论，与方向无关）；
         普通臂只报实测值，不做方向性声称——单次随机方向上它既不必然放大也不必然
         衰减，所以正文只引用射影臂的结论。
    注意不能测「权重整体乘 c」——phi(cz) = cz + λc³z³ ≠ c·phi(z)，
    三次项破坏齐次性，那不是不变性。

(B) 最小 plain 模型在 (w, v) = (0, 0) 的 Hessian：若恒为零阵，则原点是
    退化临界点（二阶判别失效），其损失恰为 Σ y²，即「输出恒为零的网络」的值。

(C) 残差臂分支零初始化（u = 0）：block 的 Jacobian 恰为单位阵，梯度无损穿过。
    这条接近同义反复，故配 (D) 才有说服力。

(D) 深度 30、无归一化残差堆的梯度保持率：分支零初始化 vs 默认初始化，
    多 seed 取比值。这才是「如何初始化」的实测依据。

跑法（限 10 核）：
  experiments/with_cpu_cap.sh python3 \
      experiments/init_probe.py
"""
import json
from pathlib import Path

import torch

torch.set_default_dtype(torch.float64)
LAM = 0.01
OUT = Path(__file__).resolve().parent / "init_probe.json"

X2 = torch.tensor([0.5, -1.2])
Y2 = torch.tensor([1.0, -0.4])
# (B) 原点 Hessian 用三个样本（与代数侧 plain m=1, n=3 同一组数），
# 这样「原点处 loss = Σy²」和 Hessian 的形状可以与 Bertini 的
# witness 点/特征值逐项对照。
X3 = torch.tensor([0.5, -1.2, 2.0])
Y3 = torch.tensor([1.0, -0.4, 0.3])


def phi(z):
    return z + LAM * z ** 3


# ------------------------------------------------- (A1) 投影前缩放不变性
def probe_projective_invariance(seed=0, width=4, depth=3):
    g = torch.Generator().manual_seed(seed)
    x = torch.tensor([0.5, -1.2])
    Ws = ([torch.randn(width, 2, generator=g)] +
          [torch.randn(width, width, generator=g) for _ in range(depth - 2)] +
          [torch.randn(1, width, generator=g)])

    def forward(scale_before_projection=1.0):
        h = x
        for W in Ws[:-1]:
            h = phi(W @ h)
            h = scale_before_projection * h        # 缩放发生在投影之前
            h = h / h.norm()                       # ℓ2 单位化
        return Ws[-1] @ h

    y_ref = forward(1.0)
    out = [{"c": c, "y": float(forward(c)),
            "absdiff": float(abs(forward(c) - y_ref))}
           for c in [1e-6, 1e-2, 1e2, 1e6]]
    return {"ref_y": float(y_ref), "scaled_before_projection": out}


# ------------------------------------------------- (A2) 层范数剖面
def probe_norm_profile(seed=0, width=32, depth=30):
    """同一初始分布下，射影臂 vs 普通臂的逐层激活范数。"""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(width, generator=g)
    x = x / x.norm()

    def wmat(i):
        W = torch.randn(width, width, generator=g) / (width ** 0.5)
        return W

    Ws = [wmat(i) for i in range(depth)]

    hn = x.clone()
    norms_norm = []
    for W in Ws:
        hn = phi(W @ hn)
        hn = hn / hn.norm()
        norms_norm.append(float(hn.norm()))

    hu = x.clone()
    norms_unnorm = []
    for W in Ws:
        hu = phi(W @ hu)
        norms_unnorm.append(float(hu.norm()))

    return {"depth": depth, "width": width,
            "projected_first_last": [norms_norm[0], norms_norm[-1]],
            "projected_all_one": all(abs(v - 1.0) < 1e-12 for v in norms_norm),
            "unnormalized_first_last": [norms_unnorm[0], norms_unnorm[-1]],
            "unnormalized_growth_factor": norms_unnorm[-1] / max(norms_unnorm[0], 1e-300)}


# ------------------------------------------------- (B) 原点 Hessian
def origin_loss_torch(t):
    """最小 plain 模型（m=1, 3 样本）的平方损失，参数向量 t = (w, v)。

    原样复用 harness 的多项式激活（clamp ±10）—— 在本例的 |w x| <= 3.6 范围内
    clamp 不生效，故与代数侧的未截断 φ(z) = z + λ z³ 逐字一致。
    """
    w, v = t[0], t[1]
    return sum((v * phi(w * x) - y) ** 2 for x, y in zip(X3, Y3))


def probe_origin_hessian():
    """(w, v) = (0, 0) 处的 loss 与 Hessian（真依赖参数，不是常数函数）。

    历史 bug：旧版写的是 `lambda t: loss()`，闭包里的 loss() 只读外层的 w, v，
    对入参 t 毫无依赖，autograd 于是返回零矩阵，得出「原点 Hessian 恒为零」的
    错误结论。这里显式让 t 进入计算。
    """
    t = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    loss0 = float(origin_loss_torch(t).detach())
    H = torch.autograd.functional.hessian(origin_loss_torch, t).detach()
    ev = torch.linalg.eigvalsh(H).detach().numpy()
    sum_y2 = float((Y3 ** 2).sum())
    # 解析闭式：H = [[2 v² Σ x² φ'(wx)², 2 Σ x φ(wx) φ'(wx)], [·, 2 Σ φ(wx)²]]
    anal = [[0.0, float(-2 * (X3 * Y3).sum())], [float(-2 * (X3 * Y3).sum()), 0.0]]
    return {"loss_at_origin": loss0, "sum_y_squared": sum_y2,
            "hessian": H.numpy().tolist(), "eigenvalues": ev.tolist(),
            "hessian_zero": bool((H.abs() < 1e-14).all()),
            "hessian_rank": int(torch.linalg.matrix_rank(H).item()),
            "hessian_indefinite": bool((ev < -1e-12).any() and (ev > 1e-12).any()),
            "hessian_analytic": anal,
            "analytic_matches": bool((H - torch.tensor(anal, dtype=torch.float64)).abs().max() < 1e-12),
            "loss_equals_sum_y_squared": bool(abs(loss0 - sum_y2) < 1e-12)}


# ------------------------------------------------- (C) 残差零初始化
def probe_residual_identity(seed=0, width=4):
    """零初始化分支时，block 的 Jacobian 恰为单位阵；分支自身的 Jacobian 恰为零阵。

    注意这条本身近乎同义反复（u=0 使 φ(uh)=0），所以必须配上 (D) 的梯度保持率
    测量才有信息量：真正要验证的是「深度 30 无归一化残差堆里，零初始化分支能否
    让梯度无损穿过」。
    """
    g = torch.Generator().manual_seed(seed)
    h1 = torch.randn(width, generator=g)
    u = torch.zeros(width, requires_grad=True)
    branch = lambda h: phi(u * h)
    Jb = torch.autograd.functional.jacobian(branch, h1)
    full = lambda h: h + branch(h)
    J = torch.autograd.functional.jacobian(full, h1)
    return {"jacobian_max_dev_from_identity": float((J - torch.eye(width)).abs().max()),
            "is_identity": bool(torch.allclose(J, torch.eye(width), atol=1e-12)),
            "branch_jacobian_norm": float(Jb.norm()),
            "branch_is_zero": bool(Jb.norm() < 1e-14)}


# ------------------------------------------------- (D) 深度残差堆的梯度保持率
# 与 harness 对齐的关键：ProjectivePolynomial.forward 先 clamp 再取三次项
# (harness.py:52-54)，所以 φ 有界；本探针若不 clamp 就会溢出，得到的 NaN
# 结论属于另一个模型。CLAMP/宽度/隐层排布都与 harness 保持一致。
CLAMP = 10.0


def probe_residual_depth_gradient(seed=0, depth=30, width=32, branch_init=0.0):
    """深度残差堆 x -> x + φ(U φ(W x))，与 harness 的 r1 臂同构（clamp ±10，无归一化）。

    branch_init 是输出侧权重 U 的初始化尺度（U = branch_init * randn/sqrt(width)）。
    恒等路径的解析参照：若分支恒为 0，则 sum(h) 对 x 的梯度是分量全 1 的向量，
    范数恰为 sqrt(width) —— 零初始化分支应当精确命中这个值，与深度无关。
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(width, generator=g)
    x = (x / x.norm()).requires_grad_(True)

    Us, Ws = [], []
    for _ in range(depth):
        U = branch_init * torch.randn(width, width, generator=g) / (width ** 0.5)
        W = torch.randn(width, width, generator=g) / (width ** 0.5)
        Us.append(U)
        Ws.append(W)

    h = x.clone()
    for U, W in zip(Us, Ws):
        h = h + _phi_clamped(U, W, h)
    y = h.sum()
    (grad,) = torch.autograd.grad(y, x)
    gn = float(grad.norm())
    return {"depth": depth, "width": width, "branch_init": branch_init,
            "grad_norm": gn, "grad_norm_identity_ref": float(width ** 0.5),
            "finite": bool(gn == gn and gn not in (float("inf"), float("-inf"))),
            "hidden_norm_final": float(h.norm()) if float(h.norm()) == float(h.norm()) else None}


def _phi_clamped(U, W, h):
    """φ(U φ(W h))，两处都按 harness 的方式 clamp 到 ±CLAMP。"""
    z1 = (W @ h).clamp(-CLAMP, CLAMP)
    a = (z1 + LAM * z1 ** 3).clamp(-CLAMP, CLAMP)
    z2 = (U @ a).clamp(-CLAMP, CLAMP)
    return (z2 + LAM * z2 ** 3).clamp(-CLAMP, CLAMP)


def main():
    res: dict = {"projective_invariance": probe_projective_invariance(),
           "norm_profile": probe_norm_profile(),
           "origin_hessian": probe_origin_hessian(),
           "residual_zero_init_identity": probe_residual_identity()}

    a = res["projective_invariance"]
    print("=== (A1) 投影前激活缩放 c 倍 ===")
    print(f"  参考输出 y = {a['ref_y']:.12f}")
    for r in a["scaled_before_projection"]:
        print(f"  c={r['c']:<9g} y={r['y']:.12f}  |Δy|={r['absdiff']:.2e}")

    n = res["norm_profile"]
    print(f"\n=== (A2) 逐层激活范数（depth={n['depth']}, width={n['width']}）===")
    print(f"  射影臂:   首={n['projected_first_last'][0]:.12f}  "
          f"末={n['projected_first_last'][1]:.12f}  每层恒为 1: {n['projected_all_one']}")
    print(f"  普通臂:   首={n['unnormalized_first_last'][0]:.4e}  "
          f"末={n['unnormalized_first_last'][1]:.4e}  "
          f"放大 {n['unnormalized_growth_factor']:.3e}x")

    o = res["origin_hessian"]
    print("\n=== (B) 最小 plain 模型原点处的 Hessian ===")
    print(f"  L(0,0) = {o['loss_at_origin']:.6f}   Σy² = {o['sum_y_squared']:.6f}"
          f"   相等: {o['loss_equals_sum_y_squared']}")
    print(f"  Hessian = {o['hessian']}")
    print(f"  特征值 = {[round(e, 12) for e in o['eigenvalues']]}"
          f"   秩 = {o['hessian_rank']}   恒为零: {o['hessian_zero']}"
          f"   不定: {o['hessian_indefinite']}")
    print(f"  解析闭式 H = {o['hessian_analytic']}   与实测一致: {o['analytic_matches']}")

    c = res["residual_zero_init_identity"]
    print("\n=== (C) 残差臂分支零初始化 ===")
    print(f"  block Jacobian 与单位阵最大偏差 = {c['jacobian_max_dev_from_identity']:.2e}"
          f"   恒等 = {c['is_identity']}")
    print(f"  分支自身 Jacobian 范数 = {c['branch_jacobian_norm']:.2e}"
          f"   恒为零 = {c['branch_is_zero']}")

    print("\n=== (D) 深度 30 残差堆的梯度范数 vs 分支初始化尺度（无归一化）===")
    seeds = [0, 1, 2]
    scales = [0.0, 0.01, 0.1, 1.0]
    rows = []
    for sc in scales:
        rs = [probe_residual_depth_gradient(seed=s, branch_init=sc) for s in seeds]
        ok = [r for r in rs if r["finite"]]
        rows.append({"branch_init": sc, "runs": rs,
                     "median_grad_norm": float(sorted(r["grad_norm"] for r in ok)[len(ok) // 2])
                     if ok else None,
                     "n_finite": len(ok), "n_total": len(rs)})
        med = rows[-1]["median_grad_norm"]
        print(f"  branch_init={sc:<6g} 有限 {len(ok)}/{len(rs)}  "
              f"梯度范数中位数 = {'NaN' if med is None else f'{med:.4e}'}")
    res["depth_gradient"] = {"depth": 30, "width": 32, "seeds": seeds, "scales": rows}
    ref = rows[0]["runs"][0]["grad_norm_identity_ref"]
    res["depth_gradient_identity_ref"] = ref
    exact = all(abs(r["grad_norm"] - ref) < 1e-9 for r in rows[0]["runs"])
    res["depth_gradient_zero_init_exact_identity"] = bool(exact)
    print(f"  恒等路径解析参照 sqrt(width) = {ref:.4f}；"
          f"零初始化三分支均精确命中: {exact}")

    OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\n[写出] {OUT}")


if __name__ == "__main__":
    main()
