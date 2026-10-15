#!/usr/bin/env python3
"""实数临界点计数（唯一数据源，供正文中「几个复解中只有 x 个是实的」这类句子）。

tab_algebraic 只报 dim I 与 |I|（复解个数）。论文正文还用到一句更强的断言：
"五个复解里只有 (w, v) = (0, 0) 是实的"。这个数字也必须由脚本给出。

做法（全精确算术，不碰浮点）：
  1. lex Groebner 消元得到只含 w 的消元多项式（最后一个变量已被消掉）；
  2. sp.real_roots() 数实根（含重数）；
  3. 交叉核对：sp.solve() 全解，按 .is_real 判实；
  4. 把实点代回原临界方程，报最大残差。

跑法（限 10 核）：
  experiments/with_cpu_cap.sh python3 \
      experiments/algebraic_real_points.py
"""
import json
import time
from pathlib import Path

import sympy as sp

LAM = sp.Rational(1, 100)
SAMPLES = [(sp.Rational(1, 2), sp.Integer(1)),
           (sp.Rational(-6, 5), sp.Rational(-2, 5)),
           (sp.Integer(2), sp.Rational(3, 10))]

OUT = Path(__file__).resolve().parent / "algebraic_real_points.json"


def phi(z, lam=LAM):
    return z + lam * z ** 3


def plain_system(m, n, lam=LAM):
    """yhat = sum_j v_j phi(w_j x)，n 个样本的 square loss 的临界方程。"""
    w = sp.symbols(f"w1:{m + 1}")
    v = sp.symbols(f"v1:{m + 1}")
    L = 0
    for x, y in SAMPLES[:n]:
        L += (sum(v[j] * phi(w[j] * x, lam) for j in range(m)) - y) ** 2
    eqs = [sp.expand(sp.diff(L, t)) for t in list(w) + list(v)]
    return list(w) + list(v), eqs


def elimination_poly(vars_, eqs, keep):
    """把 keep 之外的变量排前面，lex GB 的最后一个生成元即只含 keep 的消元式。

    变量顺序很关键：默认顺序下 GB 的最后元素只含 v，拿不到单变量 w 多项式。
    """
    keep = set(keep)
    other = [t for t in vars_ if t not in keep]
    G = sp.groebner(eqs, *(other + [t for t in vars_ if t in keep]), order="lex")
    for g in reversed(G.exprs):
        if set(g.free_symbols) <= keep:
            return sp.factor(g)
    return None


def real_and_complex_counts(m, n):
    vars_, eqs = plain_system(m, n)
    w = vars_[:m]

    # (1) 精确实数计数：消元多项式上的 real_roots（含重数）
    elim = elimination_poly(vars_, eqs, set(w))
    nreal_elim = None
    if elim is not None:
        try:
            rr = sp.Poly(elim, *w).real_roots(multiple=True)
            nreal_elim = len(rr)
        except Exception:
            nreal_elim = None

    # (2) 全解 + 判实（交叉核对），并记录实点坐标
    sols = sp.solve(eqs, vars_, dict=True)
    pts, nreal = [], 0
    for s in sols:
        if len(s) != len(vars_):
            continue  # 参数化解（正维），此配置不该有
        vals = [s.get(t, t) for t in vars_]
        if all(v.is_real for v in vals):
            nreal += 1
            pts.append([sp.nsimplify(v) for v in vals])
    return {
        "ncomplex": len(sols),
        "nreal": nreal,
        "nreal_elim": nreal_elim,
        "elimination": sp.srepr(elim) if False else sp.sstr(elim),
        "real_points": [[str(v) for v in p] for p in pts],
    }


def verify(m, n, pt):
    """把点（字符串形式的精确值）代回原始临界方程，报最大残差。"""
    vars_, eqs = plain_system(m, n)
    sub = dict(zip(vars_, [sp.sympify(c) for c in pt]))
    out = []
    for e in eqs:
        out.append(abs(sp.N(e.subs(sub), 30)))
    return max(out) if out else None


def main():
    print("=== 实数临界点计数（dim=0 的 2 变量配置）===")
    rows = []
    for (m, n) in [(1, 2), (1, 3)]:
        t0 = time.time()
        r = real_and_complex_counts(m, n)
        resid = [float(verify(m, n, p)) for p in r["real_points"]]
        dt = time.time() - t0
        row = {
            "label": f"plain-depth1-m{m}-n{n}",
            "m": m, "n": n,
            "ncomplex": r["ncomplex"],
            "nreal": r["nreal"],
            "nreal_elim": r["nreal_elim"],
            "elimination": r["elimination"],
            "real_points": r["real_points"],
            "max_residual": max(resid) if resid else None,
            "secs": round(dt, 2),
        }
        rows.append(row)
        consistent = "一致" if r["nreal"] == r["nreal_elim"] else "不一致!"
        print(f"  {row['label']:<20} 复解={r['ncomplex']}  实解={r['nreal']}  "
              f"(消元法 {r['nreal_elim']}，{consistent})")
        print(f"      消元多项式 w: {r['elimination']}")
        print(f"      实点: {r['real_points']}   残差={row['max_residual']:.2e}  ({dt:.1f}s)")

    OUT.write_text(json.dumps(rows, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\n[写出] {OUT}  ({len(rows)} 行)")


if __name__ == "__main__":
    main()
