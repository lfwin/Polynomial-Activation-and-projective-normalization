#!/usr/bin/env sage
"""
重计算部分：Groebner 基下的临界簇维数 / 次数。

只跑 sympy 撑不住、或需要 QQbar 代数数的情形：
  * 宽 2 深 2 的复合臂（残差 / 残差+射影 / 纯射影）
  * 无归一化 plain 的符号解（QQbar，用于实/复判别）
不需要 Groebner 的结构性结论与梯度边界在
experiments/algebraic_critical_points.py（普通 venv 可跑）。

跑法（务必限核）：
  experiments/with_cpu_cap.sh sage experiments/algebraic_critical_points.sage
"""
import json
import signal

from sage.all import QQ, PolynomialRing

LAM = QQ(1) / 100
OUT = "/tmp/ag_heavy.json"
RESULTS = []


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


def phi(z):
    return z + LAM * z ** 3


def stats(R, eqs, label, secs=420, extra=None):
    I = R.ideal(eqs)
    maxdeg = int(max(e.degree() for e in eqs))
    d = timed(lambda: I.dimension(), secs)
    row = {"label": label, "nvars": R.ngens(), "maxdeg": maxdeg, "dim": str(d)}
    if d == 0:
        row["degree"] = str(timed(lambda: I.vector_space_dimension(), secs))
    if extra:
        row.update(extra)
    RESULTS.append(row)
    print(f"  {label:<34} vars={R.ngens():>2}  maxdeg={maxdeg:>3}  dim={d}"
          + (f"  degree={row.get('degree')}" if d == 0 else ""))
    return row


# ---------- 复合臂：光 2、深 2，对角权重 ----------
def composite(projective, residual, width=2, x=QQ(1) / 2, y=QQ(1)):
    names = ([f'w{j}' for j in range(1, width + 1)] +
             [f'u{j}' for j in range(1, width + 1)] +
             [f's{j}' for j in range(1, width + 1)] +
             (['r'] if projective else []))
    R = PolynomialRing(QQ, names)
    g = R.gens()
    w, u, s = g[:width], g[width:2 * width], g[2 * width:3 * width]

    h1 = [phi(w[j] * x) for j in range(width)]
    h2 = ([h1[j] + phi(u[j] * h1[j]) for j in range(width)] if residual
          else [phi(u[j] * h1[j]) for j in range(width)])

    if not projective:
        L = R((sum(s[j] * h2[j] for j in range(width)) - y) ** 2)
        return R, [L.derivative(t) for t in g], L

    r = g[3 * width]
    num = sum(s[j] * h2[j] for j in range(width)) - y * r
    H = sum(hj ** 2 for hj in h2)
    Phi = r ** 2 - H
    eqs = [2 * num * num.derivative(t) * H - num ** 2 * H.derivative(t)
           for t in list(w) + list(u) + list(s)]
    eqs.append(Phi)
    eqs += [Phi * num.derivative(t) for t in list(w) + list(u) + list(s)]
    return R, [R(e) for e in eqs], (num, H, Phi)


print("=== 复合臂（宽 2、深 2、对角）：临界簇维数 ===")
for (proj, resid, label) in [(False, True, "residual R1, no norm"),
                             (True, True, "residual + projective"),
                             (True, False, "projective only")]:
    R, eqs, _ = composite(proj, resid)
    stats(R, eqs, label)

# ---------- 无归一化的符号解（实/复判别）----------
print("\n=== 无归一化 plain m=1, n=2：QQbar 解 ===")
R = PolynomialRing(QQ, ['w1', 'v1'])
w1, v1 = R.gens()
samples = [(QQ(1) / 2, QQ(1)), (QQ(-6) / 5, QQ(-2) / 5)]
L = R(sum((v1 * phi(w1 * x) - y) ** 2 for x, y in samples))
eqs = [L.derivative(w1), L.derivative(v1)]
row = stats(R, eqs, "plain m=1,n=2")
try:
    from sage.all import QQbar
    V = timed(lambda: R.ideal(eqs).variety(QQbar), 300)
    if isinstance(V, list):
        row["n_solutions"] = len(V)
        row["solutions"] = [{str(k): str(v) for k, v in p.items()} for p in V]
        print(f"    解数={len(V)}；实解={sum(all(v.imag() == 0 for v in p.values()) for p in V)}")
        for p in V:
            print("      ", {str(k): str(v) for k, v in p.items()})
except Exception as e:
    print("    variety 失败:", type(e).__name__, str(e)[:100])

json.dump(RESULTS, open(OUT, "w"), indent=1, ensure_ascii=False)
print(f"\n[写出] {OUT}")
