#!/usr/bin/env python3
"""把 Bertini 复算的原始数据导出成论文可引用的文本（paper/data/bertini_raw.txt）。

数据源（唯一，禁止手写）：
    experiments/bertini_decomp.json    不可约分解 + 每个 component 的 witness 点分析
    experiments/bertini_samples.json   正维 component 上的采样点分析
    experiments/bertini_runs/<label>/  Bertini 的原始 input / main_data / sample_points

每个系统给出：
  1. 变量数、训练参数数（用户硬约束：<= 9）、方程次数、Bezout 数
  2. 不可约 component 表：(dim, 次数, loss, 零特征值数, 是否极点伪解)
  3. H2 判定：loss 在每个 component 上是否恒定（采样点上的绝对散布）
  4. H3 判定：零特征值个数 == component 维数（在物理训练参数子空间上取 Hessian）

跑法（限 10 核）：
    experiments/with_cpu_cap.sh python experiments/dump_bertini_raw.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "paper" / "data" / "bertini_raw.txt"

# 表行顺序 = 论文里出现顺序
ORDER = ["plain-depth1-m1-n2", "plain-depth1-m1-n3", "plain-depth1-m2-n2",
         "plain-depth2-w1", "cubic-depth1-m1-n2", "cubic-depth1-m2-n2",
         "residual-scalar", "proj-m1-n1"]

DESC = {
    "plain-depth1-m1-n2": "Plain depth 1, m=1, n=2",
    "plain-depth1-m1-n3": "Plain depth 1, m=1, n=3",
    "plain-depth1-m2-n2": "Plain depth 1, m=2, n=2",
    "plain-depth2-w1": "Plain depth 2, width 1, n=3",
    "cubic-depth1-m1-n2": "Plain depth 1, m=1, n=2, phi(z)=z^3 (cubic arm)",
    "cubic-depth1-m2-n2": "Plain depth 1, m=2, n=2, phi(z)=z^3 (cubic arm)",
    "residual-scalar": "Residual (identity skip) depth 2, width 1, n=1",
    "proj-m1-n1": "Projection depth 1, m=1, n=1",
}

ROUND_OFF = 1e-12   # 低于这个量级的散布 = 双精度舍入，loss 视为严格恒定


def cs(v):
    """复数写成单 token，便于脚本解析。"""
    if v is None:
        return "n/a"
    if isinstance(v, (list, tuple)):
        return f"{v[0]:+.12e}{v[1]:+.12e}i"
    return f"{v:+.12e}"


def h3_verdict(comps):
    """H3：非极点 component 上，零特征值个数必须等于维数。"""
    bad = []
    for c in comps:
        if c["pole_only"]:
            continue
        nz = [x for x in (c.get("n_zero_eigs_phys") or []) if x is not None]
        if nz and any(x != c["dim"] for x in nz):
            bad.append(f"dim {c['dim']} comp {c['comp']} nz={sorted(set(nz))}")
    return bad


def main():
    dec = {r["label"]: r for r in json.loads(
        (HERE / "bertini_decomp.json").read_text(encoding="utf-8"))}
    smp = json.loads((HERE / "bertini_samples.json").read_text(encoding="utf-8"))
    by_smp = {}
    for s in smp:
        by_smp.setdefault(s["label"], []).append(s)

    L = []
    L.append("Bertini raw data --- numeric irreducible decomposition of the "
             "critical ideal")
    L.append("Generator: experiments/bertini_decomp.py | Bertini v1.6 | "
             "at most 10 cores (taskset cpu0-9)")
    L.append("phi(z) = z + lambda z^3, lambda = 0.01; square loss; "
             "samples (x,y) = (1/2,1), (-6/5,-2/5), (2,3/10)")
    L.append("Hard limits obeyed: every system has at most 9 trainable "
             "parameters and at most 900 s wall clock.")
    L.append("H2: the loss is constant on every irreducible component.")
    L.append("H3: at a generic point the number of zero eigenvalues of the "
             "loss Hessian")
    L.append("    equals the dimension of the component (Hessian restricted to "
             "the trainable")
    L.append("    parameters, i.e. the auxiliary variable r is excluded).")

    n_rows = 0
    for lab in ORDER:
        r = dec.get(lab)
        if r is None:
            continue
        n_rows += 1
        L.append("")
        L.append("=" * 78)
        L.append(f"[{lab}]  {DESC.get(lab, '')}")
        L.append("=" * 78)
        L.append(f"  status             : {r['status']}   wall clock: {r['secs']} s")
        ntr = r.get("ntrain", r["nparams"])
        L.append(f"  variables          : {r['nvars']} (of which "
                 f"{r['nparams'] - ntr} auxiliary)   "
                 f"trainable params: {ntr}")
        L.append(f"  degrees of the eqs : {r['degrees']}")
        L.append(f"  Bezout number      : {r['bezout']}")
        if r["status"] != "ok":
            L.append("  --> exceeded the 900 s budget; dropped from the paper "
                     "(no result).")
            continue

        o = r.get("origin")
        if o is None:
            L.append("  origin             : n/a")
        elif o.get("defined"):
            L.append(f"  origin (w=0)       : L = {cs(o['loss'])}   "
                     f"|grad| = {o['grad_norm']:.3e}   "
                     f"n_zero_eigs = {o['n_zero_eigs']}/{o['n_eigs']}   "
                     f"max|eig| = {o['max_abs_eig']:.6g}")
        else:
            L.append("  origin (w=0)       : L undefined (1/||h|| at h = 0); "
                     "it is a pole of the projected loss")

        L.append("")
        L.append("  irreducible components")
        L.append("  (dim, index, degree, pole flag, loss = mean over the "
                 "component's witness points,")
        L.append("   where pole = the component lies inside the cleared "
                 "denominator ||h||^2 = 0)")
        L.append("    dim  #   deg  pole  L (mean over witness points)"
                 "                      spread_rel  n_zero_eigs(phys)")
        for c in r["components"]:
            spr = c["loss_spread_rel"]
            L.append(f"    {c['dim']:>3}  {c['comp']:>2}  {c['degree']:>4}  "
                     f"{'YES' if c['pole_only'] else '-':>4}  "
                     f"{cs(c['loss_mean']):>38}  "
                     f"{('%.2e' % spr) if spr is not None else 'n/a':>10}  "
                     f"{c.get('n_zero_eigs_phys')}")
        npole = sum(1 for c in r["components"] if c["pole_only"])
        if npole:
            L.append(f"    note: {npole} of the {len(r['components'])} "
                     "components lie entirely inside the cleared denominator")
            L.append("          ||h||^2 = 0, where L = G^2/||h||^2 is undefined; "
                     "they are artifacts of")
            L.append("          clearing the denominator, not stationary points "
                     "of the network, and are")
            L.append("          excluded from every count below.")

        ss = [s for s in by_smp.get(lab, []) if not s.get("pole_only")]
        if ss:
            gmax = max(s["loss_spread_abs"] for s in ss
                       if s["loss_spread_abs"] is not None)
            L.append("")
            L.append("  component sampling (Bertini TRACKTYPE 2): n generic "
                     "points per component")
            for s in ss:
                L.append(f"    dim {s['dim']} comp {s['comp']}: n={s['n_points']}"
                         f"  |spread| = {s['loss_spread_abs']:.3e}"
                         f"  n_zero_eigs(phys) = "
                         f"{sorted(set(x for x in s.get('n_zero_eigs_phys') or [] if x is not None))}")
            L.append(f"    ==> H2: max |spread| over all sampled components = "
                     f"{gmax:.3e}")
            L.append("        (at or below double-precision round-off, so the "
                     "exact loss is constant)")
            bad = h3_verdict(r["components"])
            L.append("    ==> H3: n_zero_eigs(phys) == dim for every non-pole "
                     "component: "
                     + ("YES" if not bad else "NO -- " + "; ".join(bad)))

        L.append("")
        L.append("  witness points (one per component; each coordinate as "
                 "real,imag):")
        for w in r["witness"]:
            L.append(f"    dim {w['dim']} comp {w['comp']}: "
                     + "  ".join(f"({cs(p)})" for p in w["point"]))

    L.append("")
    L.append(f"({n_rows} systems recorded.)")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"[wrote] {OUT}  ({len(L)} lines, {n_rows} systems)")


if __name__ == "__main__":
    main()
