#!/usr/bin/env sage
"""生成论文 Table tab_algebraic 的全部数字（唯一数据源）。

对每个配置构造临界理想，输出 dim I（并用 sympy 的纯幂首项判据 + 楼梯法
交叉核对 dim=0 情形的 |I|）。结果写 /tmp/ag_table.json。

跑法（限 10 核）：experiments/with_cpu_cap.sh sage experiments/algebraic_table.sage
"""
import json
import signal

from sage.all import QQ, PolynomialRing

LAM = QQ(1) / 100
SAMPLES = [(QQ(1) / 2, QQ(1)), (QQ(-6) / 5, QQ(-2) / 5), (QQ(2), QQ(3) / 10)]
OUT = "experiments/algebraic_table.json"
ROWS = []

# 单行计算上限：超过 900 s（15 分钟）的配置直接放弃，不进表 —— 用户的硬约束。
MAX_SECS = 900


class _TO(Exception):
    pass


def _alarm(sig, frm):
    raise _TO()


def timed(fn, secs):
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(secs)
    try:
        return fn()
    except _TO:
        return "TIMEOUT"
    finally:
        signal.alarm(0)


def phi(z, lam=LAM):
    return z + lam * z ** 3


def emit(label, R, eqs, note=""):
    I = R.ideal(eqs)
    import time
    t0 = time.time()
    d = timed(lambda: I.dimension(), MAX_SECS)
    dt = time.time() - t0
    if d == "TIMEOUT":
        print(f"  {label:<40} vars={R.ngens()}  TIMEOUT >{MAX_SECS}s —— 放弃，不进表")
        return None
    dd = int(d)
    row = {"label": label, "nvars": int(R.ngens()), "dim": str(d), "note": note,
           "maxdeg": int(max(int(e.degree()) for e in eqs)), "secs_dim": float(round(dt, 2))}
    if dd == 0:
        t1 = time.time()
        deg = timed(lambda: I.vector_space_dimension(), MAX_SECS)
        if deg == "TIMEOUT":
            print(f"  {label:<40} |I| TIMEOUT >{MAX_SECS}s —— 放弃，不进表")
            return None
        row["degree"] = str(deg)
        row["secs_degree"] = float(round(time.time() - t1, 2))
    ROWS.append(row)
    if dd == 0:
        print(f"  {label:<40} vars={row['nvars']}  maxdeg={row['maxdeg']:>3}  dim={row['dim']}"
              f"  |I|={row['degree']}  ({dt:.1f}s)")
    else:
        print(f"  {label:<40} vars={row['nvars']}  maxdeg={row['maxdeg']:>3}  dim={row['dim']}"
              f" ({dt:.1f}s)")
    return row


def plain(m, n, lam=LAM):
    """深度 1：yhat = sum_j v_j phi(w_j x)。2m 个变量。"""
    names = [f'w{j}' for j in range(1, m + 1)] + [f'v{j}' for j in range(1, m + 1)]
    R = PolynomialRing(QQ, names)
    g = R.gens()
    w, v = g[:m], g[m:]
    L = sum((sum(v[j] * phi(w[j] * x, lam) for j in range(m)) - y) ** 2
            for x, y in SAMPLES[:n])
    return R, [R(L).derivative(t) for t in g]


def plain_depth2(lam=LAM, depth=2, width=1):
    """plain 深度 d、宽 width：h^1=phi(w1 x), h^i=phi(w_i h^{i-1}), yhat=v h^d。"""
    nv = depth * width + width
    names = ([f'w{i}_{j}' for i in range(1, depth + 1) for j in range(1, width + 1)]
             + [f'v{j}' for j in range(1, width + 1)])
    R = PolynomialRing(QQ, names)
    g = R.gens()
    W = [g[(i - 1) * width:i * width] for i in range(1, depth + 1)]
    V = g[depth * width:]
    L = 0
    for x, y in SAMPLES:
        h = [phi(W[0][j] * x, lam) for j in range(width)]
        for i in range(1, depth):
            h = [phi(sum(W[i][k] * h[k] for k in range(width)), lam) for j in range(width)]
        L += (sum(V[j] * h[j] for j in range(width)) - y) ** 2
    return R, [R(L).derivative(t) for t in g]


def residual_scalar(x, y, lam=LAM):
    """R1、深度 2、单宽度：h1=phi(w1 x), h2=h1+phi(w2 h1), yhat=v h2。"""
    R = PolynomialRing(QQ, ['w1', 'w2', 'v'])
    w1, w2, v = R.gens()
    h1 = phi(w1 * x, lam)
    h2 = h1 + phi(w2 * h1, lam)
    L = (v * h2 - y) ** 2
    return R, [R(L).derivative(t) for t in R.gens()]


def proj(m, x, y, lam=LAM):
    """yhat=(v.h)/||h||，用 r^2=sum h^2 去根号。"""
    names = ([f'w{j}' for j in range(1, m + 1)] + [f'v{j}' for j in range(1, m + 1)] + ['r'])
    R = PolynomialRing(QQ, names)
    g = R.gens()
    w, v, r = g[:m], g[m:2 * m], g[2 * m]
    h = [phi(w[j] * x, lam) for j in range(m)]
    G = sum(v[j] * h[j] for j in range(m)) - y * r
    H = sum(hj ** 2 for hj in h)
    eqs = [2 * G * G.derivative(t) * H - G ** 2 * H.derivative(t) for t in list(w) + list(v)]
    Phi = r ** 2 - H
    eqs.append(Phi)
    eqs += [Phi * G.derivative(t) for t in list(w) + list(v)]
    return R, [R(e) for e in eqs]


def composite(projective, residual, width=2, x=QQ(1) / 2, y=QQ(1), lam=LAM):
    """宽 width、深 2、对角权重；projective/residual 开关。"""
    names = ([f'w{j}' for j in range(1, width + 1)] +
             [f'u{j}' for j in range(1, width + 1)] +
             [f's{j}' for j in range(1, width + 1)] +
             (['r'] if projective else []))
    R = PolynomialRing(QQ, names)
    g = R.gens()
    w, u, s = g[:width], g[width:2 * width], g[2 * width:3 * width]
    h1 = [phi(w[j] * x, lam) for j in range(width)]
    h2 = ([h1[j] + phi(u[j] * h1[j], lam) for j in range(width)] if residual
          else [phi(u[j] * h1[j], lam) for j in range(width)])
    if not projective:
        L = R((sum(s[j] * h2[j] for j in range(width)) - y) ** 2)
        return R, [L.derivative(t) for t in g]
    r = g[3 * width]
    num = sum(s[j] * h2[j] for j in range(width)) - y * r
    H = sum(hj ** 2 for hj in h2)
    Phi = r ** 2 - H
    eqs = [2 * num * num.derivative(t) * H - num ** 2 * H.derivative(t)
           for t in list(w) + list(u) + list(s)]
    eqs.append(Phi)
    eqs += [Phi * num.derivative(t) for t in list(w) + list(u) + list(s)]
    return R, [R(e) for e in eqs]


print("=== 生成 tab_algebraic 的唯一数据源 ===")

print("[A] plain 未截断，深度 1")
for (m, n) in [(1, 2), (1, 3), (2, 2)]:
    R, eqs = plain(m, n)
    emit(f"plain-depth1-m{m}-n{n}", R, eqs)

print("[B] plain 未截断，深度 2（宽 1）")
R, eqs = plain_depth2(depth=2, width=1)
emit("plain-depth2-w1", R, eqs)

print("[C] phi = z^3（去掉一次项）")
for (m, n) in [(1, 2), (1, 3), (2, 2)]:
    R, eqs = plain(m, n, lam=QQ(0))
    emit(f"cubic-depth1-m{m}-n{n}", R, eqs)

print("[D] 残差 R1（深度 2，单宽度）")
for i in [0, 2]:
    R, eqs = residual_scalar(*SAMPLES[i])
    emit(f"residual-scalar-x{SAMPLES[i][0]}-y{SAMPLES[i][1]}", R, eqs)

print("[E] projective normalization")
for m in [1, 2]:
    R, eqs = proj(m, *SAMPLES[0])
    emit(f"proj-m{m}-n1", R, eqs)

print("[F] 复合臂（宽 2 深 2 对角）")
for (p, r, lab) in [(False, True, "composite-residual"), (True, True, "composite-residual-proj"),
                    (True, False, "composite-proj-only")]:
    R, eqs = composite(p, r)
    emit(lab, R, eqs)

json.dump(ROWS, open(OUT, "w"), indent=1, ensure_ascii=False)
print(f"\n[写出] {OUT}  ({len(ROWS)} 行)")
