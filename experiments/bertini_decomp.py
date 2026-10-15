#!/usr/bin/env python
"""Bertini（数值代数几何）验证前作 arXiv:2102.04338 的两个猜想 + 原始数据落盘。

要验证的命题
------------
H2  **loss 在每个不可约 component 上恒定**（前作 Hypothesis 1/2 的核心）。
    推论 Corollary 1：交于原点的若干 component 共享同一 loss。
H3  **generic 点处 Hessian 的常零特征值个数 == 该 component 的维数**（前作 Hypothesis 3）；
    推论 Corollary 2：任意点处零特征值个数 >= 该 component 维数（交点处更大）。

方法（与 Bertini 官方三件套一一对应）
------------------------------------
1. numerical irreducible decomposition（TRACKTYPE 1）→ 每个 component 的 dim/degree + witness set；
2. component sampling（TRACKTYPE 2）→ 在每个正维 component 上采 k 个 generic 点，
   算 loss 散布（验 H2）与 Hessian 零特征值个数（验 H3）；
3. 原点：直接算 loss / Hessian，验 Corollary 1/2。

原始数据全部写入 experiments/bertini_decomp.json 与 experiments/bertini_samples.json
（witness 点全精度坐标、loss、Hessian 特征值），供 make_macros.py 生成论文宏。

硬约束（用户）
-------------
* 训练参数 <= 9（本脚本 <= 7）；* 计算 <= 10 核（taskset 0-9，串行）；
* 单个系统 <= 900 s，超时即放弃、不进表（MAX_SECS）。
* 变量数、维数、loss、特征值一律来自本脚本实测，禁止手写。

跑法
----
  experiments/with_cpu_cap.sh .venv/bin/python experiments/bertini_decomp.py
  ... --only proj-m1-n1        # 只跑一个
  ... --hard                   # 只跑超预算的两个（900s 上限，验是否真弃）
"""
import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import sympy as sp

# ---------------------------------------------------------------- 模型定义
LAM = sp.Rational(1, 100)
SAMPLES = [(sp.Rational(1, 2), sp.Rational(1)),
           (sp.Rational(-6, 5), sp.Rational(-2, 5)),
           (sp.Rational(2), sp.Rational(3, 10))]

MAX_SECS = 900
BERTINI = os.environ.get("BERTINI", "bertini")
HERE = Path(__file__).resolve().parent
AUX = "r"          # 唯一辅助变量名（射影臂去根号用）；不算训练参数
RUNAWAY = HERE / "bertini_runs"
OUT_DECOMP = HERE / "bertini_decomp.json"
OUT_SAMPLES = HERE / "bertini_samples.json"


def phi(z):
    return z + LAM * z ** 3


# ---------------------------------------------------------------- 系统构造
def _h1(m, x, w, bit):
    """bit = phi(w x) 的截面。"""
    return [z + LAM * z ** 3 for z in (w[j] * x for j in range(m))]


def _plain_depth1(m, n, lam=LAM):
    """深度 1：yhat = sum_j v_j phi(w_j x)。2m 个变量，2m 个训练参数。"""
    w = sp.symbols(f"w1:{m + 1}")
    v = sp.symbols(f"v1:{m + 1}")
    L = sum((sum(v[j] * (w[j] * x + lam * (w[j] * x) ** 3) for j in range(m)) - y) ** 2
            for x, y in SAMPLES[:n])
    vs = list(w) + list(v)
    return [str(t) for t in vs], [sp.expand(sp.diff(L, t)) for t in vs], \
        [str(t) for t in vs], L, None


def _plain_depth2(w1n, w2n, v):
    """深度 2 单宽度 plain：h1=phi(w1 x), h2=phi(w2 h1), yhat=v h2（3 样本）。"""
    x1, y1 = SAMPLES[0]
    L = 0
    for x, y in SAMPLES:
        h1 = w1n * x + LAM * (w1n * x) ** 3
        h2 = w2n * h1 + LAM * (w2n * h1) ** 3
        L += (v * h2 - y) ** 2
    vs = [w1n, w2n, v]
    return [str(t) for t in vs], [sp.expand(sp.diff(L, t)) for t in vs], \
        [str(t) for t in vs], L, None


def sys_plain_depth2_w1():
    w1, w2, v = sp.symbols("w1 w2 v")
    return _plain_depth2(w1, w2, v)


def _residual_scalar(x, y):
    """R1 深度 2 单宽度：h1=phi(w1 x), h2=h1+phi(w2 h1), yhat=v h2（单样本）。"""
    w1, w2, v = sp.symbols("w1 w2 v")
    h1 = w1 * x + LAM * (w1 * x) ** 3
    h2 = h1 + (w2 * h1 + LAM * (w2 * h1) ** 3)
    L = (v * h2 - y) ** 2
    vs = [w1, w2, v]
    return [str(t) for t in vs], [sp.expand(sp.diff(L, t)) for t in vs], \
        [str(t) for t in vs], L, None


def sys_residual_scalar():
    return _residual_scalar(*SAMPLES[0])


def _proj(m, x, y):
    """yhat = <v,h>/||h||；用 r^2 = ||h||^2 去根号。

    临界方程 F_t = 2 G G_t H - G^2 H_t = 0（G = <v,h> - y r, H = <h,h>），
    连同 Phi = r^2 - H = 0，共 2m+1 方程 / 2m+1 变量（方系统）。
    """
    w = sp.symbols(f"w1:{m + 1}")
    v = sp.symbols(f"v1:{m + 1}")
    r = sp.symbols("r")
    h = [w[j] * x + LAM * (w[j] * x) ** 3 for j in range(m)]
    Hh = sum(hj ** 2 for hj in h)
    G = sum(v[j] * h[j] for j in range(m)) - y * r
    eqs = [sp.expand(2 * G * sp.diff(G, t) * Hh - G ** 2 * sp.diff(Hh, t))
           for t in list(w) + list(v)]
    eqs.append(sp.expand(r ** 2 - Hh))
    names = [str(s) for s in list(w) + list(v) + [r]]
    # 关键：射影臂的 loss 含 1/||h||，不是多项式，直接开根号会让复数 witness 点上的
    # 取值依赖 sqrt 的分支。代数上正确的写法是把 ||h|| = r 当作变量带进簇内
    # （r² = ||h||² 已作为方程），于是 yhat = <v,h>/r, L = (<v,h>/r - y)² = (G/r)² ——
    # 在每个 component 上都是良定义的代数函数，「loss 在 component 上恒定」才可判定。
    # 注意：探针变量必须与临界方程组一致 —— 方程组是在 (w, v, r) 扩展空间上写的
    # （r² = ||h||² 已作为方程带入），所以 Hessian 也在同一空间上取，否则 Lnet 里的
    # r 无法代入，lambdify 会退回符号表达式。
    Lnet = (G / r) ** 2
    # 第 5 项：分母 Hh = ||h||²。清去分母后簇里会出现 Hh ≡ 0 的「伪 component」
    # （整支落在极点 ‖h‖=0 上，L = G²/Hh 在那里无定义）。带上它才能在后面把
    # 这些伪解识别出来并排除，否则会误读成「loss 在 component 上不恒定」。
    return (names, eqs, [str(s) for s in list(w) + list(v) + [r]], Lnet, Hh)


SYSTEMS = {
    "plain-depth1-m1-n2": lambda: _plain_depth1(1, 2),
    "plain-depth1-m1-n3": lambda: _plain_depth1(1, 3),
    "plain-depth1-m2-n2": lambda: _plain_depth1(2, 2),
    "plain-depth2-w1": sys_plain_depth2_w1,
    "cubic-depth1-m1-n2": lambda: _plain_depth1(1, 2, lam=sp.Rational(0)),
    "cubic-depth1-m2-n2": lambda: _plain_depth1(2, 2, lam=sp.Rational(0)),
    "residual-scalar": sys_residual_scalar,
    "proj-m1-n1": lambda: _proj(1, *SAMPLES[0]),
    "proj-m2-n1": lambda: _proj(2, *SAMPLES[0]),
}

# 先跑这些（Bézout 可承受）
FEASIBLE = ["plain-depth1-m1-n2", "plain-depth1-m1-n3", "plain-depth1-m2-n2",
            "cubic-depth1-m1-n2", "cubic-depth1-m2-n2", "proj-m1-n1"]
# 超预算嫌疑（Bézout >= 1.5e4），单独用 900 s 上限确认是否真该弃
HARD = ["plain-depth2-w1", "residual-scalar", "proj-m2-n1"]


# ---------------------------------------------------------------- Bertini 驱动
def bertini_expr(e):
    """sympy 表达式 -> Bertini 语法（** -> ^）。"""
    return str(sp.expand(e)).replace("**", "^")


# 固定随机种子。Bertini 文档（RANDOMSEED）：设为 0 时每次运行取不同的随机数，
# 于是同一个系统的不可约分解会随运行而变 —— 即使系统完全确定。实测 proj-m1-n1
# 的 component 个数在多次运行中在 4 和 5 之间跳。论文要求可复现，故写死。
RANDOM_SEED = 20261014


def write_input(path, names, eqs, tracktype=1):
    lines = ["CONFIG", "", f"TRACKTYPE: {tracktype};",
             f"RANDOMSEED: {RANDOM_SEED};", "", "END;", "", "INPUT", ""]
    lines.append("variable_group " + ", ".join(names) + ";")
    fnames = [f"f{i + 1}" for i in range(len(eqs))]
    lines.append("function " + ", ".join(fnames) + ";")
    for fn, e in zip(fnames, eqs):
        lines.append(f"{fn} = {bertini_expr(e)};")
    lines += ["", "END;", ""]
    text = "\n".join(lines)
    path.write_text(text, encoding="utf-8")
    return text


def _clean_env():
    env = dict(os.environ)
    for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY",
              "all_proxy", "ALL_PROXY", "LD_LIBRARY_PATH"):
        env.pop(k, None)
    return env


def _run(cwd, feed=None, secs=MAX_SECS):
    """跑 Bertini（<=10 核）。超时后连子进程组一起收，避免悬挂。"""
    try:
        p = subprocess.run(["taskset", "-c", "0-9", BERTINI], cwd=str(cwd),
                           input=feed, capture_output=True, text=True,
                           timeout=secs, env=_clean_env(), start_new_session=True)
        return ("ok" if p.returncode == 0 else f"rc={p.returncode}"), p.stdout, p.stderr
    except subprocess.TimeoutExpired as ex:
        out = (ex.stdout or b"")
        if isinstance(out, bytes):
            out = out.decode("utf-8", "replace")
        return f"TIMEOUT >{secs}s", out, ""


DIM_RE = re.compile(r"-{10}DIMENSION\s+(\d+)-{10}")
COMP_RE = re.compile(r"Component number:\s*(\d+)")


def parse_main_data(txt, nvars):
    """按 (dim, comp) 分组解析 main_data 的 witness 点（坐标行连续出现）。"""
    out = []
    blocks = DIM_RE.split(txt)          # [pre, dim, body, dim, body, ...]
    for i in range(1, len(blocks), 2):
        d = int(blocks[i])
        body = blocks[i + 1].split("***************")[0]
        for chunk in body.split("---------------"):
            if "Path number" not in chunk:
                continue
            mc = COMP_RE.search(chunk)
            if not mc:
                continue
            coords = []
            for ln in chunk.splitlines():
                parts = ln.split()
                if len(parts) != 2:
                    continue
                try:
                    coords.append(complex(float(parts[0]), float(parts[1])))
                except ValueError:
                    continue
            if len(coords) < nvars:
                continue
            for k in range(0, len(coords) - nvars + 1, nvars):
                out.append({"dim": d, "comp": int(mc.group(1)),
                            "point": [[z.real, z.imag] for z in coords[k:k + nvars]]})
    return out


def total_degrees(names, eqs):
    syms = sp.symbols(names)
    degs = []
    for e in eqs:
        p = sp.Poly(sp.expand(e), *syms)
        degs.append(int(p.total_degree()))
    return degs


# ---------------------------------------------------------------- loss / Hessian
def make_funs(pnames, Lnet):
    """解析 loss / 梯度 / Hessian（evaluate 用 numpy 复数）。"""
    args = sp.symbols(pnames)
    fL = sp.lambdify(args, Lnet, modules="numpy")
    grads = [sp.diff(Lnet, t) for t in args]
    fg = [sp.lambdify(args, g, modules="numpy") for g in grads]
    fh = [[sp.lambdify(args, sp.diff(g, t2), modules="numpy") for t2 in args]
          for g in grads]

    def loss(p):
        return complex(np.asarray(fL(*p), dtype=complex))

    def grad(p):
        return np.array([complex(np.asarray(f(*p), dtype=complex)) for f in fg], dtype=complex)

    def hessian(p):
        n = len(p)
        H = np.zeros((n, n), dtype=complex)
        for i in range(n):
            for j in range(n):
                H[i, j] = complex(np.asarray(fh[i][j](*p), dtype=complex))
        return H

    return loss, grad, hessian


def probe_point(loss, grad, hessian, pfull, pnames, names, den=None, phys_idx=None):
    """对一点算 loss / |grad| / Hessian 谱，并给出零特征值个数（相对阈值 1e-6）。

    某些模型的 loss 在原点本身无定义（如射影臂的 1/||h||，h=0 处 0/0），
    此时标记 defined=False 而不是让 NaN 污染整张表。den 给出被清掉的分母
    （射影臂的 Hh = ||h||²）：den ~ 0 说明该点落在极点集合上，属于清分母时
    人为引入的伪解，记 pole=True（残差量级 |den| 一并记下，便于审计）。
    """
    idx = {n: i for i, n in enumerate(names)}
    p = np.array([pfull[idx[n]] for n in pnames], dtype=complex)
    dv = None
    if den is not None:
        with np.errstate(all="ignore"):
            dv = complex(np.asarray(den(*p), dtype=complex))
    with np.errstate(all="ignore"):
        L, g, H = loss(p), grad(p), hessian(p)
    finite = (np.isfinite(L.real) and np.isfinite(L.imag)
              and bool(np.all(np.isfinite(H))) and bool(np.all(np.isfinite(g))))
    pole = bool(dv is not None and abs(dv) < 1e-8)
    ev = np.linalg.eigvals(H) if (finite and len(p)) else np.array([])
    if not finite or not np.all(np.isfinite(ev)):
        return {"params": [[z.real, z.imag] for z in p], "defined": False,
                "pole": pole, "den_abs": None if dv is None else abs(dv),
                "loss": None, "grad_norm": None, "n_eigs": 0,
                "n_zero_eigs": None, "n_zero_eigs_phys": None,
                "max_abs_eig": None, "eigs": []}
    mx = float(np.max(np.abs(ev))) if len(ev) else 0.0
    thr = 1e-6 * max(mx, 1e-12)
    # 物理训练参数子块（对射影臂就是 (w, v)，把辅助变量 r 排除在外）。
    # H3 说的「Hessian 的零特征值个数」是训练参数方向上的，辅助变量不该混进来。
    nz_phys = None
    if phys_idx:
        Hp = H[np.ix_(phys_idx, phys_idx)]
        evp = np.linalg.eigvals(Hp)
        if np.all(np.isfinite(evp)):
            mxp = float(np.max(np.abs(evp))) if len(evp) else 0.0
            nz_phys = int(np.sum(np.abs(evp) < 1e-6 * max(mxp, 1e-12)))
    return {
        "params": [[z.real, z.imag] for z in p],
        "defined": True,
        "pole": pole,
        "den_abs": None if dv is None else abs(dv),
        "loss": [L.real, L.imag],
        "grad_norm": float(np.linalg.norm(g)),
        "n_eigs": len(ev),
        "n_zero_eigs": int(np.sum(np.abs(ev) < thr)),
        "n_zero_eigs_phys": nz_phys,
        "max_abs_eig": mx,
        "eigs": [[z.real, z.imag] for z in ev],
    }


# ---------------------------------------------------------------- 主流程
def decompose(label, spec, secs=MAX_SECS):
    names, eqs, pnames, Lnet, _den = spec
    d = RUNAWAY / label
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    inp = write_input(d / "input", names, eqs, 1)
    t0 = time.time()
    status, sout, serr = _run(d, feed=None, secs=secs)
    el = round(time.time() - t0, 2)
    wit = parse_main_data((d / "main_data").read_text(errors="replace")
                          if (d / "main_data").exists() else "", len(names))
    return {"label": label, "status": status, "secs": el, "nvars": len(names),
            "nparams": len(pnames),
            # 真正的「训练参数」个数：排除辅助变量 r（射影臂去根号用），
            # 用户的硬约束「最多 9 个训练参数」判的是这个数。
            "ntrain": len([n for n in pnames if n != AUX]),
            "degrees": total_degrees(names, eqs),
            "bezout": math.prod(total_degrees(names, eqs)),
            "input": inp, "workdir": str(d), "witness": wit,
            "stdout_tail": sout[-1500:]}


def sample_component(label, spec, dim, comp, npoints):
    names, eqs, pnames, Lnet, _den = spec
    d = RUNAWAY / label
    inp = (d / "input").read_text(encoding="utf-8")
    (d / "input").write_text(re.sub(r"TRACKTYPE:\s*1\s*;", "TRACKTYPE: 2;", inp),
                             encoding="utf-8")
    (d / "sample_points").unlink(missing_ok=True)
    feed = f"{dim}\n{comp}\n{npoints}\n0\n"
    t0 = time.time()
    status, sout, _ = _run(d, feed=feed)
    el = round(time.time() - t0, 2)
    (d / "input").write_text(inp, encoding="utf-8")
    pts = []
    sp_path = d / "sample_points"
    if sp_path.exists():
        blk, cur = [], []
        for ln in sp_path.read_text(errors="replace").splitlines()[1:]:
            parts = ln.split()
            if len(parts) != 2:
                if cur:
                    blk.append(cur)
                    cur = []
                continue
            try:
                cur.append(complex(float(parts[0]), float(parts[1])))
            except ValueError:
                continue
        if cur:
            blk.append(cur)
        pts = [b for b in blk if len(b) == len(names)]
    _den = spec[4]
    _denf = (sp.lambdify(sp.symbols(names), _den, modules="numpy")
             if _den is not None else None)
    loss, grad, hessian = make_funs(pnames, Lnet)
    phys = [i for i, n in enumerate(pnames) if n != AUX]
    rows = [probe_point(loss, grad, hessian, pt, pnames, names, _denf, phys)
            for pt in pts]
    fin = [r for r in rows if r.get("defined") and not r.get("pole")]
    lv = np.array([complex(*r["loss"]) for r in fin]) if fin else np.array([])
    return {"label": label, "dim": dim, "comp": comp, "n_requested": npoints,
            "pole_only": bool(rows) and all(r.get("pole") for r in rows),
            "status": status, "secs": el, "n_points": len(rows),
            "n_probed": len(fin),
            "loss_mean": ([lv.mean().real, lv.mean().imag] if len(lv) else None),
            "loss_spread_abs": (float(np.max(np.abs(lv - lv.mean())))
                                if len(lv) else None),
            "n_zero_eigs": [r["n_zero_eigs"] for r in fin],
            "n_zero_eigs_phys": [r.get("n_zero_eigs_phys") for r in fin],
            "points": rows}


def load(path):
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return []
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", action="append", default=None)
    ap.add_argument("--hard", action="store_true", help="只跑超预算嫌疑组（900 s 上限）")
    ap.add_argument("--nsample", type=int, default=5, help="每个正维 component 采样点数")
    args = ap.parse_args()

    labels = args.only or (HARD if args.hard else FEASIBLE)
    decomp = [r for r in load(OUT_DECOMP) if not any(l == r["label"] for l in labels)]
    samples = [r for r in load(OUT_SAMPLES) if not any(l == r["label"] for l in labels)]

    for lab in labels:
        spec = SYSTEMS[lab]()
        names, eqs, pnames, Lnet, _den = spec
        print(f"\n=== {lab}: vars={len(names)} eqs={len(eqs)} params={len(pnames)} "
              f"degrees={total_degrees(names, eqs)}", flush=True)
        rec = decompose(lab, spec)
        print(f"  decomposition: {rec['status']} in {rec['secs']}s  "
              f"Bézout={rec['bezout']}", flush=True)
        if rec["status"] != "ok":
            decomp.append({"label": lab, "status": rec["status"], "secs": rec["secs"],
                           "nvars": rec["nvars"], "nparams": rec["nparams"],
                           "ntrain": rec["ntrain"],
                           "degrees": rec["degrees"], "bezout": rec["bezout"],
                           "input": rec["input"], "witness": [], "origin": None,
                           "components": []})
            OUT_DECOMP.write_text(json.dumps(decomp, indent=1), encoding="utf-8")
            continue

        _denf = (sp.lambdify(sp.symbols(names), _den, modules="numpy")
                 if _den is not None else None)
        loss, grad, hessian = make_funs(pnames, Lnet)
        phys = [i for i, n in enumerate(pnames) if n != AUX]
        groups = {}
        for w in rec["witness"]:
            pfull = [complex(a, b) for a, b in w["point"]]
            groups.setdefault((w["dim"], w["comp"]), []).append(
                probe_point(loss, grad, hessian, pfull, pnames, names, _denf,
                            phys))
        comps = []
        for (dim, comp), rows in sorted(groups.items()):
            # 极点伪 component：整个 component 的分母恒为 0（清分母时人为引入）
            pole_only = bool(rows) and all(r.get("pole") for r in rows)
            fin = [r for r in rows if r.get("defined") and not r.get("pole")]
            if fin:
                lv = np.array([complex(*r["loss"]) for r in fin])
                scale = max(float(np.max(np.abs(lv))), 1e-30)
                spread = float(np.max(np.abs(lv - lv.mean())))
                lmean = [lv.mean().real, lv.mean().imag]
                # 数值零：witness 点上的 L 是 ~1e-32 的舍入噪声（真值恰为 0），
                # 此时「相对散布」会被这个噪声放大成 O(1) 的假信号。用绝对地板
                # 1e-24（远高于双精度噪声、远低于任何真实 loss）把它归零。
                lspr = 0.0 if scale < 1e-24 else spread / scale
            else:                      # 该 component 的全部 witness 点都不可评估
                lv, lmean, spread, lspr = np.array([]), None, None, None
            comps.append({"dim": dim, "comp": comp, "degree": len(rows),
                          "n_probed": len(fin), "pole_only": pole_only,
                          "loss_mean": lmean, "loss_spread_abs": spread,
                          "loss_spread_rel": lspr,
                          "n_zero_eigs": [r["n_zero_eigs"] for r in rows],
                          "n_zero_eigs_phys": [r.get("n_zero_eigs_phys")
                                               for r in rows],
                          "n_zero_eigs_probed": [r["n_zero_eigs"] for r in fin],
                          "points": rows})
        o = probe_point(loss, grad, hessian, [0j] * len(names), pnames, names, _denf)
        rec.update({"origin": o, "components": comps})
        decomp.append(rec)
        OUT_DECOMP.write_text(json.dumps(decomp, indent=1), encoding="utf-8")
        for c in comps:
            lm = complex(*c["loss_mean"]) if c["loss_mean"] else None
            sp_ = f"{c['loss_spread_rel']:.2e}" if c["loss_spread_rel"] is not None else "n/a"
            tag = "  [POLE: 清分母伪解，已排除]" if c["pole_only"] else ""
            lstr = f"{complex(*c['loss_mean']):.8g}" if c["loss_mean"] else "n/a"
            print(f"    dim {c['dim']} comp {c['comp']}: deg={c['degree']} "
                  f"loss={lstr} spread_rel={sp_} "
                  f"nzero={c['n_zero_eigs']}{tag}", flush=True)
        if o.get("defined"):
            print(f"    origin: loss={complex(*o['loss']):.8g} "
                  f"nzero={o['n_zero_eigs']}/{o['n_eigs']}", flush=True)
        else:
            print("    origin: loss undefined (e.g. 1/||h|| at h=0)", flush=True)

        # 正维 component 采样（H2 的真检验）；同一 dim 下 component 号可能重复但我们只用唯一对
        for (dim, comp_id) in sorted({(c["dim"], c["comp"]) for c in comps}):
            if dim <= 0:
                continue
            s = sample_component(lab, spec, dim, comp_id, args.nsample)
            samples.append(s)
            OUT_SAMPLES.write_text(json.dumps(samples, indent=1), encoding="utf-8")
            print(f"    sample dim {dim} comp {comp_id}: {s['status']} "
                  f"{s['secs']}s  n={s['n_points']}  "
                  f"spread_abs={s['loss_spread_abs']}  nzero={s['n_zero_eigs']}",
                  flush=True)

    print(f"\n[写出] {OUT_DECOMP} ({len(decomp)} 条) / {OUT_SAMPLES} ({len(samples)} 条)")


if __name__ == "__main__":
    main()
