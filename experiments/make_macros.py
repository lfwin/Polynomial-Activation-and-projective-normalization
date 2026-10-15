#!/usr/bin/env python3
"""
把实验 JSON 汇总成 paper/macros.tex。

正文里的每一个数字都通过 \\MacroName 引用，宏由此脚本生成，
从而保证论文数字与真实实验输出一致，不会有手写/臆造的数字。

激活集合与 experiments/run_all.py 严格一致：
  逐元素: identity, tanh, relu, gelu, silu, poly, poly_proj   (H=128)
  门控:   swiglu, geglu, polyglu                              (H=82)

用法:
  python experiments/make_macros.py --exp-dir experiments --out paper/macros.tex
  python experiments/make_macros.py --check     # 校验论文引用的宏是否都已定义
"""
import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np

# 必须与 run_all.py 保持一致
ELEM = ["identity", "tanh", "relu", "gelu", "silu", "poly", "poly_proj"]
GATED = ["swiglu", "geglu", "polyglu"]
GATED_H = 82
ELEM_H = 128


def hid_for(act):
    return GATED_H if act in GATED else ELEM_H


# 独立参考组：这些组跑的是与主网格相同的激活/norm/depth/hidden/seed，但
# epoch 预算不同（ref_orig=20ep, ref_fidelity=2ep）。它们与 10-epoch 网格
# 共用同一个配置 key，若混在同一批记录里，任何未过滤 epochs 的查询都可能
# 取到它们（例如 \polyProjAcc 会从 10ep 的 95.93 被顶成 20ep 的 96.95）。
# 因此默认把它们排除在分析总体之外，需要时用 out="ref_orig" 显式点名。
_SOLO_GROUPS = {"ref_orig", "ref_fidelity", "residual_mnist"}


def _is_solo(r):
    p = r.get("_path")
    return p is not None and Path(p).parent.name in _SOLO_GROUPS


def load(exp_dir):
    recs = []
    for f in glob.glob(str(Path(exp_dir) / "*" / "*.json")):
        try:
            with open(f, encoding="utf-8") as fh:
                r = json.load(fh)
                r["_path"] = f          # 记录来源，供 out= 组限定使用
                recs.append(r)
        except Exception:
            pass
    return recs


def pick(recs, **q):
    # out="ref_orig" -> 只看该组；否则默认排除独立参考组。
    want = q.pop("out", None)
    # 归一化位置默认只取 post：norm_mnist 组里 layer/rms 同时有 pre 与 post
    # 两条记录，pick() 返回列表、one() 取 m[0]，不钉死位置就会随 glob 顺序
    # 在 96.29（post）与 95.09（pre）之间抖动。需要 pre 时显式传 norm_pos="pre"。
    if "norm_pos" not in q:
        q["norm_pos"] = "post"
    out = []
    for r in recs:
        if want is not None:
            p = r.get("_path")
            if p is None or Path(p).parent.name != want:
                continue
        elif _is_solo(r):
            continue
        a = r["args"]
        if all(a.get(k) == v for k, v in q.items()):
            out.append(r)
    return out


def one(recs, **q):
    m = pick(recs, **q)
    return m[0] if m else None


def acc2(v):
    """准确率格式化为两位小数字符串。"""
    return f"{v * 100:.2f}"


def sci(x):
    """把浮点写成 LaTeX 科学计数法：3.16e-16 -> 3.16\\times10^{-16}。"""
    if x == 0:
        return "0"
    e = int(np.floor(np.log10(abs(x))))
    m = x / 10.0 ** e
    return rf"{m:.2f}\times10^{{{e}}}"


# LaTeX 控制序列名只能由字母组成，宏名里绝不能出现数字，
# 否则 \newcommand{\L20PolyAcc} 会被 TeX 解析成 \L + 文本，编译报错。
_N2W = {2: "two", 3: "three", 4: "four", 5: "five", 10: "ten",
        15: "fifteen", 20: "twenty", 30: "thirty", 50: "fifty",
        100: "hundred", 200: "twohundred", 300: "threehundred"}


def n2w(n):
    """数字 -> 英文单词，用于构造合法的宏名。"""
    return _N2W.get(n, "n" + str(n).replace("0", "o").replace("1", "i"))


# lambda 取值 -> 纯字母词，用于宏名（宏名里绝不能出现数字或小数点）
_LAM2W = {0.001: "OneThousandth", 0.01: "OneHundredth",
          0.05: "FiveHundredths", 0.1: "OneTenth"}


def lam2w(lam):
    """把 lambda 的数值转成纯字母词，供宏名拼接。"""
    key = round(float(lam), 6)
    if key in _LAM2W:
        return _LAM2W[key]
    return "Lam" + "".join(c for c in f"{key:g}" if c.isalpha())


def build(recs):
    M = {}

    def s(k, v):
        M[k] = v

    # ---------- 参考网络（严格复现原脚本：poly_proj, 无归一化, 20 epochs）----------
    r = one(recs, out="ref_orig", dataset="mnist", activation="poly_proj",
            norm="none", depth=10, hidden=128, seed=42, epochs=20)
    if r:
        s("refTestAcc", acc2(r["test_accuracy"]))
        s("refTrainAcc", acc2(r["train_accuracy"]))
        s("refParams", f"{r['n_params']:,}")
        s("refEpochSec", f"{r['seconds_per_epoch']:.1f}")
        s("refTotalSec", f"{r['seconds']:.0f}")
        s("refEpochs", str(r["args"]["epochs"]))
        h = r["history"]
        if h.get("train_acc"):
            s("refEoneTrain", acc2(h["train_acc"][0]))
            s("refEtwoTrain", acc2(h["train_acc"][1]))
        if h.get("test_acc"):
            s("refEoneTest", acc2(h["test_acc"][0]))
            s("refEtwoTest", acc2(h["test_acc"][1]))

    # 超参宏必须无条件定义：它们描述的是 harness 的固定设置，不依赖 ref_orig
    # 记录是否已跑完。否则生成器可能不产出 \lrBase/\wdecay，pdflatex 直接报
    # Undefined control sequence。
    s("polyLam", "0.01")
    s("polyDeg", "3")
    s("lrBase", "0.001")
    s("wdecay", "0.0001")

    # ---------- 保真度校验（2-epoch 运行，与原脚本同配置）----------
    rf = one(recs, out="ref_fidelity", dataset="mnist", activation="poly_proj",
             norm="none", depth=10, hidden=128, seed=42, epochs=2)
    if rf:
        hf = rf["history"]
        # 宏名要与论文一致：\fidEOneTrain / \fidEOneTest / \fidETwoTrain / ...
        # 注意前缀已经含 fid，"fidE" + "One" + "Train"，不要把 macro 再写成
        # "fidTrain"（那样会生成 fidEOnefidTrain，论文引用不到）。
        for key, macro in (("train_acc", "Train"), ("test_acc", "Test")):
            seq = hf.get(key) or []
            for i, v in enumerate(seq[:2], 1):
                word = {1: "One", 2: "Two"}[i]
                s(f"fidE{word}{macro}", acc2(v))

    s("nCfgs", str(len(recs)))

    # ---------- 门控宽度对齐的诚实记账 ----------
    # W_e=128 与 W_g=82 的参数量差：论文原文手写 "within 0.4%" 是错的，
    # 这里按真实记录算出来（逐元素 233866 / 门控 238466 -> 约 2.0%）。
    # 必须限定组目录：poly_proj/none 同时在 act_mnist/width_mnist/ref_* 里出现，
    # poly_proj 又和 width_mnist 里的 poly/l2 等价，泛查询可能取到别的组。
    try:
        pe = one(recs, out="act_mnist", dataset="mnist", activation="poly_proj",
                 norm="none", depth=10, hidden=ELEM_H, seed=42, epochs=10)
        pg = one(recs, out="glu_norm_mnist", dataset="mnist",
                 activation="swiglu", norm="none", depth=10, hidden=GATED_H,
                 seed=42, epochs=10)
        if pe and pg:
            ne, ng = pe["n_params"], pg["n_params"]
            s("paramElem", f"{ne:,}")
            s("paramGate", f"{ng:,}")
            s("paramGapPct", "%.1f" % (100.0 * abs(ne - ng) / ne))
    except Exception:
        pass

    # 可学习激活相对固定多项式的训练时间倍数
    rfix = one(recs, out="act_mnist", dataset="mnist", activation="poly_proj",
               norm="none", depth=10, hidden=ELEM_H, seed=42, epochs=10)
    rlp = one(recs, out="learn_poly_mnist", dataset="mnist",
              activation="legendre", norm="none", depth=10, hidden=ELEM_H,
              seed=42, epochs=10)
    if rfix and rlp:
        s("learnCycles",
          f"{rlp['seconds_per_epoch'] / max(rfix['seconds_per_epoch'], 1e-9):.1f}")

    # ---------- 阶数网格（poly + l2，限定 degree_mnist）----------
    # 网格同时扫 degree 与 lam，两组因素量级差很多：lam 从 0.001 到 0.1
    # 带来约 1.5 点的跨度，degree 从 2 到 5 在同一 lam 下只差不到 0.3 点。
    # 因此论文的结论必须落在 lam 上，不能写成"低阶多项式更差"（不成立）。
    degvals = []
    for deg in [2, 3, 4, 5]:
        r = one(recs, out="degree_mnist", dataset="mnist", activation="poly",
                degree=deg, lam=0.01, norm="l2", depth=10, hidden=ELEM_H,
                seed=42, epochs=10)
        if r:
            s(f"deg{n2w(deg)}Acc", acc2(r["test_accuracy"]))
            degvals.append((deg, r["test_accuracy"]))
        for lam in [0.001, 0.01, 0.05, 0.1]:
            c = one(recs, out="degree_mnist", dataset="mnist",
                    activation="poly", degree=deg, lam=lam, norm="l2",
                    depth=10, hidden=ELEM_H, seed=42, epochs=10)
            if c:
                # lam 也必须转成词：宏名不能含数字，否则 \degfiveLam0p001Acc
                # 会被 TeX 读成 \degfiveLam + "0p001Acc"，报 Missing number。
                s(f"deg{n2w(deg)}Lam{lam2w(lam)}Acc", acc2(c["test_accuracy"]))
    if degvals:
        s("degTwoMinAcc", f"{min(v for _, v in degvals) * 100:.2f}")
        hi = [v for d, v in degvals if d >= 3]
        if hi:
            s("degThreeMinAcc", f"{max(hi) * 100:.2f}")
    # 全网格：lam 的跨度 vs degree 的跨度（论文用来说明哪一维才是主导）
    grid = pick(recs, out="degree_mnist", dataset="mnist", activation="poly",
                norm="l2", depth=10, hidden=ELEM_H, seed=42, epochs=10)
    if grid:
        s("degGridMinAcc", acc2(min(x["test_accuracy"] for x in grid)))
        s("degGridMaxAcc", acc2(max(x["test_accuracy"] for x in grid)))
        s("degGridBestLam",
          f"{max(grid, key=lambda x: x['test_accuracy'])['args']['lam']:g}")
        lo = [x["test_accuracy"] for x in grid if x["args"]["lam"] == 0.001]
        hi2 = [x["test_accuracy"] for x in grid if x["args"]["lam"] == 0.1]
        if lo and hi2:
            s("degLamSpanPt", f"{(np.mean(hi2) - np.mean(lo)) * 100:.2f}")
        d2 = [x["test_accuracy"] for x in grid if x["args"]["degree"] == 2]
        d5 = [x["test_accuracy"] for x in grid if x["args"]["degree"] == 5]
        if d2 and d5:
            s("degFixLamSpanPt",
              f"{abs(np.mean(d2) - np.mean(d5)) * 100:.2f}")

    # ---------- 激活函数表（MNIST, L10, 无归一化）----------
    # 必须限定 out="act_mnist"：depth_mnist/degree_mnist 等组里也有 poly/l2 之类
    # 的同名配置，泛查询会取到别的组的值（曾把 \LtenPolyprojAcc 顶成 97.36）。
    name2macro = {
        "identity": "identityNoneAcc", "tanh": "tanhNoneAcc",
        "relu": "reluNoneAcc", "gelu": "geluNoneAcc", "silu": "siluNoneAcc",
        "poly": "polyNoneAcc", "poly_proj": "polyProjAcc",
    }
    for a, k in name2macro.items():
        r = one(recs, out="act_mnist", dataset="mnist", activation=a,
                norm="none", depth=10, hidden=ELEM_H, seed=42, epochs=10)
        if r:
            s(k, acc2(r["test_accuracy"]))

    # 非饱和逐元素激活的带宽（relu/gelu/silu）
    band = []
    for a in ["relu", "gelu", "silu"]:
        r = one(recs, out="act_mnist", dataset="mnist", activation=a,
                norm="none", depth=10, hidden=ELEM_H, seed=42, epochs=10)
        if r:
            band.append(r["test_accuracy"])
    if band:
        s("actBandPt", f"{(max(band) - min(band)) * 100:.2f}")

    # 门控（无归一化）——真实门控宽度 H=82
    glu_none = []
    for a in GATED:
        r = one(recs, out="glu_norm_mnist", dataset="mnist", activation=a,
                norm="none", depth=10, hidden=GATED_H, seed=42, epochs=10)
        if r:
            glu_none.append(r["test_accuracy"])
    if glu_none:
        s("gluNoneMean", acc2(float(np.mean(glu_none))))
        s("gluNoneMax", acc2(float(np.max(glu_none))))
        s("nGluNone", str(len(glu_none)))

    # 门控 + LayerNorm 平均
    glu_ln = []
    for a in GATED:
        r = one(recs, out="glu_norm_mnist", dataset="mnist", activation=a,
                norm="layer", depth=10, hidden=GATED_H, seed=42, epochs=10)
        if r:
            glu_ln.append(r["test_accuracy"])
    if glu_ln:
        s("gluLayerMean", acc2(float(np.mean(glu_ln))))
        # 门控在无归一化下整体崩掉（激活范数 0 -> 梯度 0），配归一化后恢复。
        # 这个对比是论文的核心失败模式证据，单独建宏。
        s("gluGapPt", f"{(np.mean(glu_ln) - np.mean(glu_none)) * 100:.1f}")

    # ---------- 多项式：射影 vs 不射影（核心对比，取 act_mnist 组）----------
    rp = one(recs, out="act_mnist", dataset="mnist", activation="poly_proj",
             norm="none", depth=10, hidden=ELEM_H, seed=42, epochs=10)
    rn = one(recs, out="act_mnist", dataset="mnist", activation="poly",
             norm="none", depth=10, hidden=ELEM_H, seed=42, epochs=10)
    if rp and rn:
        s("polyGap", f"{(rp['test_accuracy'] - rn['test_accuracy']) * 100:.2f}")

    # 多项式与同层最优现代激活（GELU）的差距 —— 摘要里的 headline 数字
    best_elem = max(
        [x["test_accuracy"] for x in
         pick(recs, out="act_mnist", dataset="mnist", norm="none", depth=10,
              hidden=ELEM_H, seed=42, epochs=10)
         if x["args"]["activation"] in ("relu", "gelu", "silu")] or [0])
    if rn and best_elem:
        s("polyNoneToBest", f"{(best_elem - rn['test_accuracy']) * 100:.2f}")
    if rp and best_elem:
        s("polyProjToBest", f"{(best_elem - rp['test_accuracy']) * 100:.2f}")

    # 深度数字的英文词（宏名/正文里以词出现，避免数字宏名）
    # 注意：不能叫 \thirty / \fifty —— 那是 plain TeX 的计数寄存器别名
    # （\thirty=\count30），覆盖它们会触发 "Missing number, treated as zero"。
    for n, w in ((30, "dThirty"), (50, "dFifty")):
        s(w, str(n))

    # ---------- 可学习多项式（限定 learn_poly_mnist 组）----------
    # 可学习族内部还有超参（B-spline 的网格大小 G4/G8/G16、Chebyshev 的
    # 基函数个数），同族多条记录是同一配置族的扫描，因此取族内最优作为
    # 该族的代表成绩；固定多项式 poly 则无此自由度。
    for a, kl, kb in [("legendre", "legendreNoneAcc", "legendreBestAcc"),
                      ("chebyshev", "chebyNoneAcc", "chebyBestAcc"),
                      ("bspline", "bsplineNoneAcc", "bsplineBestAcc")]:
        fam = pick(recs, out="learn_poly_mnist", dataset="mnist",
                   activation=a, depth=10, hidden=ELEM_H, seed=42, epochs=10)
        # 无归一化 + 配归一化的最优（前者反映裸可学习多项式的失效风险，
        # 后者反映配归一化后的真实上限）
        none_v = [x["test_accuracy"] for x in fam if x["args"]["norm"] == "none"]
        all_v = [x["test_accuracy"] for x in fam]
        if none_v:
            s(kl, acc2(float(np.max(none_v))))
        if all_v:
            s(kb, acc2(float(np.max(all_v))))

    # ---------- 归一化：poly 与 relu ----------
    # 限定 out="norm_mnist"：poly/l2/layer/batch 等配置在 depth_mnist、
    # degree_mnist 等组里也大量出现，泛查询会取到别的深度的记录。
    for act in ["poly", "relu"]:
        pref = "poly" if act == "poly" else "relu"
        for norm, lbl in [("none", "None"), ("l2", "Ltwo"), ("batch", "Batch"),
                          ("layer", "Layer"), ("rms", "Rms")]:
            r = one(recs, out="norm_mnist", dataset="mnist", activation=act,
                    norm=norm, depth=10, hidden=ELEM_H, seed=42, epochs=10)
            if r:
                s(f"{pref}{lbl}Acc", acc2(r["test_accuracy"]))
    # 射影归一化对两种激活的效应差（论文的关键不对称证据）
    # poly：none -> l2 有增益；ReLU：none -> l2 略降。符号相反，量级差一个
    # 数量级，因此不能把 l2 当成与激活函数无关的即插即用归一化。
    for act, pref in (("poly", "poly"), ("relu", "relu")):
        rn_ = one(recs, out="norm_mnist", dataset="mnist", activation=act,
                  norm="none", depth=10, hidden=ELEM_H, seed=42, epochs=10)
        rl_ = one(recs, out="norm_mnist", dataset="mnist", activation=act,
                  norm="l2", depth=10, hidden=ELEM_H, seed=42, epochs=10)
        if rn_ and rl_:
            s(f"{pref}LtwoDeltaPt",
              f"{(rl_['test_accuracy'] - rn_['test_accuracy']) * 100:+.2f}")

    # ---------- 深度 ----------
    for d in [2, 3, 5, 10, 15, 20]:
        for act, norm, lbl in [("relu", "none", "Relu"),
                               ("poly", "l2", "Polyproj"),
                               ("poly", "batch", "Polybn"),
                               ("poly", "layer", "Polyln"),
                               ("poly", "none", "Poly"),
                               ("identity", "none", "Identity"),
                               ("gelu", "none", "Gelu")]:
            r = one(recs, out="depth_mnist", dataset="mnist", activation=act,
                    norm=norm, depth=d, hidden=ELEM_H, seed=42, epochs=10)
            if r:
                s(f"L{n2w(d)}{lbl}Acc", acc2(r["test_accuracy"]))

    # ---------- 极深网络：射影化的适用边界（depth_very 组）----------
    # L30/L50 是论文必须如实报告的下界：射影多项式在 10--20 层能与 ReLU
    # 抗衡（L10 差 2.14 点、L20 差 2.22 点），到 L30 差距收窄到 0.29 点，
    # 而 L50 两种设计一起塌陷（poly 11.35 / relu 11.35）。把这个边界写进
    # 论文比笼统宣称"更稳"更有说服力，也避免读者推断我们能跑到任意深度。
    for d in [30, 50]:
        for act, norm, lbl in [("relu", "none", "Relu"),
                               ("relu", "batch", "ReluBn"),
                               ("poly", "l2", "Polyproj"),
                               ("poly", "layer", "Polyln"),
                               ("poly", "none", "Poly"),
                               ("identity", "none", "Identity")]:
            r = one(recs, out="depth_very", dataset="mnist", activation=act,
                    norm=norm, depth=d, hidden=ELEM_H, seed=42, epochs=10)
            if r:
                s(f"L{n2w(d)}{lbl}Acc", acc2(r["test_accuracy"]))
    # 深网里射影多项式与 ReLU 的差距（符号为正表示 ReLU 领先）
    for d in [10, 20, 30, 50]:
        grp = "depth_mnist" if d <= 20 else "depth_very"
        rp_ = one(recs, out=grp, dataset="mnist", activation="poly", norm="l2",
                  depth=d, hidden=ELEM_H, seed=42, epochs=10)
        rr_ = one(recs, out=grp, dataset="mnist", activation="relu", norm="none",
                  depth=d, hidden=ELEM_H, seed=42, epochs=10)
        if rp_ and rr_:
            s(f"L{n2w(d)}ProjVsReluPt",
              f"{(rr_['test_accuracy'] - rp_['test_accuracy']) * 100:.2f}")
    # L50 的塌陷强度：polynomial 与 relu 都掉到随机水平（10 类 -> 10%）
    r50 = one(recs, out="depth_very", dataset="mnist", activation="poly",
              norm="none", depth=50, hidden=ELEM_H, seed=42, epochs=10)
    if r50:
        s("LfiftyCollapseAcc", acc2(r50["test_accuracy"]))

    # ---------- 恒等残差：无归一化的深层网络（residual_mnist 组）----------
    # 这一组回答一个独立于射影设计的问题：把归一化整个拿掉，只给多项式
    # 堆叠加一条恒等路径，网络还能不能在 L=100 以上训练。答案是能，而且
    # 100--300 层基本平。所有查询都必须显式 out="residual_mnist"（该组在
    # _SOLO_GROUPS 里，不会混进主网格）。
    def _res(res, initm, d, seed=None):
        q = dict(out="residual_mnist", dataset="mnist", activation="poly",
                 norm="none", depth=d, hidden=ELEM_H, residual=res,
                 init_mode=initm, epochs=10)
        if seed is not None:
            q["seed"] = seed
        return one(recs, **q)

    def _ms(res, initm, d):
        """4 seed 的均值与样本标准差（%，ddof=1）。"""
        vals = [r["test_accuracy"] for s in [42, 43, 44, 45]
                if (r := _res(res, initm, d, s)) is not None]
        if not vals:
            return None
        return (float(np.mean(vals)) * 100,
                float(np.std(vals, ddof=1) * 100), len(vals))

    # L100 主对照：4 种（残差 × 初始化）组合
    for res, initm, lbl in [("none", "xavier", "ResNoneXavier"),
                            ("none", "id", "ResNoneId"),
                            ("r1", "xavier", "ResRoneXavier"),
                            ("r1", "id", "ResRoneId")]:
        st = _ms(res, initm, 100)
        if st:
            base = f"residual{lbl}"
            s(f"{base}Mean", f"{st[0]:.2f}")
            s(f"{base}Std", f"{st[1]:.2f}")
            s(f"{base}N", str(st[2]))
    # 无残差两组的崩溃幅度（与随机水平 10% 的差）
    for initm, lbl in [("xavier", "ResNoneXavier"), ("id", "ResNoneId")]:
        st = _ms("none", initm, 100)
        if st:
            s(f"residual{lbl}GapPt", f"{st[0] - 10.0:+.2f}")
    # 残差带来的提升（不带符号，文本里读作 "gains of X and Y points" / "X points below"）
    for initm, lbl in [("xavier", "Xavier"), ("id", "Id")]:
        a = _ms("none", initm, 100)
        b = _ms("r1", initm, 100)
        if a and b:
            s(f"residual{lbl}GainPt", f"{b[0] - a[0]:.2f}")
    # 深度扫描（xavier）：与 Table~\ref{tab:residualdepth} 逐字一致 —— 每个深度都用
    # seed 42/43 两个种子，且先换算成百分数再取均值（否则 10.815 的浮点表示会让
    # 正文 10.81、表格 10.82）。
    def _sweep_mean(res, d):
        vals = [r["test_accuracy"] * 100 for s_ in (42, 43)
                if (r := _res(res, "xavier", d, s_)) is not None]
        return float(np.mean(vals)) if vals else None

    for d in [10, 50, 200, 300]:
        for res, lbl in [("none", "ResNone"), ("r1", "ResRone")]:
            m = _sweep_mean(res, d)
            if m is not None:
                s(f"residual{lbl}L{n2w(d)}Acc", f"{m:.2f}")
    # 最深的残差位置：L300 的准确率，以及它相对 L50 的衰减（文本里说的是
    # "sixfold increase in depth"，L50→L300 正好 6 倍，必须用同一对深度）。
    m300, m50 = _sweep_mean("r1", 300), _sweep_mean("r1", 50)
    if m300 is not None and m50 is not None:
        s("residualDeepestDepth", "300")
        s("residualDeepestAcc", f"{m300:.2f}")
        s("residualDeepDropPt", f"{m50 - m300:+.2f}")
    # ReLU + 残差的对照（同样无归一化）
    relu_vals = [r["test_accuracy"] for sfx in (42, 43)
                 if (r := one(recs, out="residual_mnist", dataset="mnist",
                              activation="relu", norm="none", depth=100,
                              hidden=ELEM_H, residual="r1",
                              init_mode="xavier", seed=sfx, epochs=10))
                 is not None]
    if relu_vals:
        s("residualReluRoneMean", f"{np.mean(relu_vals) * 100:.2f}")
        s("residualReluRoneN", str(len(relu_vals)))
    # 每个残差配置的末层激活范数（说明残差是在"范数自由增长"下工作的）
    rlast = _res("r1", "xavier", 100, 42)
    if rlast and rlast.get("act_norm_end"):
        s("residualRoneActNormLast", f"{rlast['act_norm_end'][-1]:.1f}")
    rnone = _res("none", "xavier", 100, 42)
    if rnone and rnone.get("act_norm_end"):
        s("residualNoneActNormLast", f"{rnone['act_norm_end'][-1]:.1f}")

    # 崩溃的机制性证据：饱和上界与预测类别数。
    # 截断立方 φ(z)=clamp(z,±c)+λ clamp(z,±c)³ 在 |z|≥c 时逐单元取到常量
    # 20 = c + λc³（c=10, λ=0.01）；若整层都饱和且方向一致，隐层范数就是
    # 20√H = 226.27。实测无残差组恰好落在这个数上 → 是"纯尺度爆炸 + 逐单元
    # 饱和"，不是射影把方向压没（本设定根本没有归一化）。
    sat = 20.0 * float(ELEM_H) ** 0.5
    s("residualSatBound", f"{sat:.2f}")

    def _ncls(res, initm):
        out_ = []
        for sd in [42, 43, 44, 45]:
            r = _res(res, initm, 100, sd)
            if r and r.get("confusion"):
                C = np.asarray(r["confusion"])
                out_.append(int((C.sum(0) > 0).sum()))
        return out_

    nx, ni = _ncls("none", "xavier"), _ncls("none", "id")
    rx = _ncls("r1", "xavier")
    if nx:
        s("residualResNoneXavierNcls", str(max(nx)))
    if ni:
        s("residualResNoneIdNcls", str(max(ni)))
        s("residualResNoneIdNclsMin", str(min(ni)))
    if rx:
        s("residualResRoneXavierNcls", str(min(rx)))
    # 无残差组的末层激活范数（4 seed 平均），与饱和上界对照
    for res, initm, lbl in [("none", "xavier", "ResNoneXavier"),
                            ("none", "id", "ResNoneId"),
                            ("r1", "xavier", "ResRoneXavier"),
                            ("r1", "id", "ResRoneId")]:
        vals = [(r["act_norm_end"][-1]) for sd in [42, 43, 44, 45]
                if (r := _res(res, initm, 100, sd)) is not None
                and r.get("act_norm_end")]
        if vals:
            s(f"residual{lbl}ActNormLast", f"{np.mean(vals):.1f}")

    # ---------- 多 seed ----------
    for lbl, q in [("Relu", dict(activation="relu", norm="none")),
                   ("Polyproj", dict(activation="poly", norm="l2")),
                   ("Polyln", dict(activation="poly", norm="layer")),
                   ("Polybn", dict(activation="poly", norm="batch"))]:
        vals = [x["test_accuracy"] for x in
                pick(recs, out="seeds", dataset="mnist", depth=10,
                     hidden=ELEM_H, **q)]
        if len(vals) > 1:
            s(f"seeds{lbl}Mean", acc2(float(np.mean(vals))))
            s(f"seeds{lbl}Std", f"{np.std(vals, ddof=1) * 100:.2f}")
            s(f"seeds{lbl}N", str(len(vals)))

    # ---------- 激活范数比（poly / relu / swiglu）----------
    for lbl, q, h in [("Poly", dict(activation="poly", norm="l2"), ELEM_H),
                      ("Relu", dict(activation="relu", norm="none"), ELEM_H),
                      ("Swi", dict(activation="swiglu", norm="none"), GATED_H)]:
        r = one(recs, out="norm_mnist" if lbl != "Swi" else "glu_norm_mnist",
                dataset="mnist", depth=10, hidden=h, seed=42, epochs=10, **q)
        if r and r.get("act_norm_init"):
            an = r["act_norm_init"]
            s(f"actNorm{lbl}First", f"{an[0]:.2f}")
            s(f"actNorm{lbl}Last", f"{an[-1]:.4f}")
            s(f"actNorm{lbl}Ratio", f"{an[-1] / max(an[0], 1e-12):.4g}")

    # ---------- 合成数据（3 层，H=64）----------
    synth_acts = ["identity", "relu", "gelu", "silu", "tanh", "poly",
                  "swiglu", "geglu"]
    for ds in ["rings", "spiral", "xor", "checker", "cubes", "parity",
               "blobs", "gauss_highdim"]:
        vals = {}
        for a in synth_acts:
            r = one(recs, out="synth3", dataset="synth_train", synth_name=ds,
                    activation=a, depth=3)
            if r:
                vals[a] = r["test_accuracy"]
        if vals:
            best = max(vals, key=vals.get)
            # LaTeX 宏名只能含字母：数据集名如 gauss_highdim 含下划线，会让
            # \newcommand{\synthGauss_highdimBest} 被解析成 \synthGauss + 下标，
            # 报 "Missing $ inserted" / "You already have nine parameters"。
            ds_tag = re.sub(r"[^A-Za-z]", "", ds).capitalize()
            s(f"synth{ds_tag}Best", f"{vals[best] * 100:.2f}")
            s(f"synth{ds_tag}BestName",
              {"poly": "polynomial", "poly_proj": "projected polynomial",
               "swiglu": "SwiGLU", "geglu": "GeGLU"}.get(best, best))
            if "relu" in vals:
                s(f"synth{ds_tag}Relu", acc2(vals["relu"]))

    # ---------- 小块损失曲面的关键点（loss_surface.json）----------
    # 数据源：experiments/loss_surface.py 的数值输出（多起点优化 + autograd 梯度范数）。
    # 注意宏名只能含字母，所以 d1/d2/d3 写成 one/two/three。
    ls_path = Path(__file__).resolve().parent / "loss_surface.json"
    if ls_path.exists():
        ls = json.loads(ls_path.read_text(encoding="utf-8"))
        for key, tag in [("d1", "One"), ("d2", "Two"), ("d3", "Three")]:
            blk = ls.get(key)
            if not blk:
                continue

            name_map = {"t0": "CritZero", "crit": "Crit", "origin": "Crit",
                        "min": "Min", "tmin": "Min", "inf": "Inf"}
            words = ["", "Two", "Three", "Four", "Five", "Six"]
            seen = {}
            for p in blk["points"]:
                # 极小点（is_min）优先归到 Min，其余按名字归类
                base = "Min" if p.get("is_min") else name_map.get(p["name"], "Crit")
                seen[base] = seen.get(base, 0) + 1
                # 宏名只能含字母，重复出现时用词缀而非数字
                pname = base if seen[base] == 1 else f"{base}{words[seen[base] - 1]}"
                s(f"lossSurf{tag}{pname}Loss", f"{p['loss']:.4f}")
                g = p.get("gradnorm")
                s(f"lossSurf{tag}{pname}Grad", "---" if g is None else sci(g))
                c = [v for v in p["coord"] if v is not None]
                s(f"lossSurf{tag}{pname}Coord",
                  ", ".join(f"{v:.4f}" for v in c) if c else "---")
        # 便于正文直接引用的可读量
        d1 = ls["d1"]
        s("lossSurfOneCritCount", str(len(d1["critical"])))
        crs = sorted(c for c in d1["critical"] if c != 0.0)
        s("lossSurfOneCritPos", f"{crs[-1]:.4f}" if crs else "---")
        s("lossSurfOneCritNeg", f"{crs[0]:.4f}" if crs else "---")
        s("lossSurfOneCritCountNonzero", str(len(crs)))
        z = next(p for p in d1["points"] if abs(p["coord"][0]) < 1e-12)
        s("lossSurfOneZeroLoss", f"{z['loss']:.4f}")
        mn = next(p for p in d1["points"] if p.get("is_min"))
        s("lossSurfOneMinLoss", f"{mn['loss']:.4f}")
        s("lossSurfTwoInfLoss", f"{ls['d2']['points'][1]['loss']:.4f}")
        s("lossSurfTwoCritLoss", f"{ls['d2']['points'][0]['loss']:.4f}")
        s("lossSurfTwoOriginGrad", sci(ls['d2']['points'][0]['gradnorm']))
        s("lossSurfThreeMinWone", f"{ls['d3']['points'][1]['coord'][0]:.4f}")
        s("lossSurfThreeMinWtwo", f"{ls['d3']['points'][1]['coord'][1]:.4f}")
        s("lossSurfThreeMinV", f"{ls['d3']['points'][1]['coord'][2]:.4f}")
        tw = ls.get("twosample")
        if tw:
            s("lossSurfTwoSampleResidual", sci(tw["residual"]))

    # ---------- 代数几何侧的宏（tab_algebraic 正文引用 + 实数点 + 初始化探针） ----------
    # 数据源：experiments/algebraic_table.json（dim/degree）、
    #         experiments/algebraic_real_points.json（实数临界点个数）、
    #         experiments/init_probe.json（初始化断言）。
    # 正文里的 dim I / |I| / 「几个复解里几个是实的」/ 初始化结论全部引用这里的宏。
    _exp = Path(__file__).resolve().parent

    alg_path = _exp / "algebraic_table.json"
    if alg_path.exists():
        alg = {r["label"]: r for r in json.loads(alg_path.read_text(encoding="utf-8"))}
        # dim / degree 逐行宏：宏名只能含字母，标签里的数字改写为词缀
        for lab, tag in [("plain-depth1-m1-n2", "PlainMoneNtwo"),
                         ("plain-depth1-m1-n3", "PlainMoneNthree"),
                         ("plain-depth1-m2-n2", "PlainMtwoNtwo"),
                         ("plain-depth2-w1", "PlainDepthTwo"),
                         ("residual-scalar-x1/2-y1", "ResidualScalar"),
                         ("composite-residual", "CompositeResidualOnly"),
                         ("composite-proj-only", "CompositeProjOnly"),
                         ("composite-residual-proj", "CompositeResidualProj")]:
            r = alg.get(lab)
            if r:
                s(f"alg{tag}Dim", str(r["dim"]))
                s(f"alg{tag}Vars", str(r["nvars"]))
                deg = r.get("degree")
                s(f"alg{tag}Degree", "---" if deg in (None, "0") else str(deg))
        # 最常引用的两个数
        if "plain-depth1-m1-n2" in alg:
            s("algPlainDim", str(alg["plain-depth1-m1-n2"]["dim"]))
            s("algPlainDegree", str(alg["plain-depth1-m1-n2"]["degree"]))
        if "plain-depth1-m1-n3" in alg:
            s("algPlainNthreeDegree", str(alg["plain-depth1-m1-n3"]["degree"]))
        if "composite-residual" in alg:
            s("algCompositeResidualDim", str(alg["composite-residual"]["dim"]))
        if "composite-proj-only" in alg:
            s("algCompositeProjDim", str(alg["composite-proj-only"]["dim"]))
        if "composite-residual-proj" in alg:
            s("algCompositeBothDim", str(alg["composite-residual-proj"]["dim"]))

    real_path = _exp / "algebraic_real_points.json"
    if real_path.exists():
        rp = {r["label"]: r for r in json.loads(real_path.read_text(encoding="utf-8"))}
        r = rp.get("plain-depth1-m1-n2")
        if r:
            s("algPlainNreal", str(r["nreal"]))
            s("algPlainNcomplex", str(r["ncomplex"]))
        r = rp.get("plain-depth1-m1-n3")
        if r:
            s("algPlainNthreeNreal", str(r["nreal"]))
            s("algPlainNthreeNcomplex", str(r["ncomplex"]))

    probe_path = _exp / "init_probe.json"
    if probe_path.exists():
        pr = json.loads(probe_path.read_text(encoding="utf-8"))
        inv = pr["projective_invariance"]
        worst = max(r["absdiff"] for r in inv["scaled_before_projection"])
        s("initProjInvarianceMaxDev", sci(worst))
        s("initProjInvarianceLevels", str(len(inv["scaled_before_projection"])))
        np_ = pr["norm_profile"]
        s("initNormProjectedLast", f"{np_['projected_first_last'][1]:.4f}")
        o = pr["origin_hessian"]
        s("initOriginLoss", f"{o['loss_at_origin']:.4f}")
        s("initOriginSumYSquared", f"{o['sum_y_squared']:.4f}")
        s("initOriginHessianRank", str(o["hessian_rank"]))
        s("initOriginHessianNonzero", "1" if not o.get("hessian_zero") else "0")
        s("initOriginHessianIndefinite", "1" if o.get("hessian_indefinite") else "0")
        ev = sorted(o["eigenvalues"])
        s("initOriginHessianEigMax", f"{ev[-1]:.4f}")
        s("initOriginHessianEigMin", f"{ev[0]:.4f}")
        s("initOriginHessianOffDiag", f"{abs(o['hessian'][0][1]):.4f}")
        s("initOriginHessianAnalyticMatches", "1" if o.get("analytic_matches") else "0")
        cz = pr["residual_zero_init_identity"]
        s("initResidualJacDev", sci(cz["jacobian_max_dev_from_identity"]))
        s("initResidualBranchJacNorm", sci(cz["branch_jacobian_norm"]))
        dg = pr.get("depth_gradient")
        if dg:
            s("initGradDepth", str(dg["depth"]))
            s("initGradWidth", str(dg["width"]))
            s("initGradIdentityRef", f"{pr['depth_gradient_identity_ref']:.4f}")
            s("initGradZeroInitExact", "1" if pr["depth_gradient_zero_init_exact_identity"] else "0")
            for row in dg["scales"]:
                sc = row["branch_init"]
                tag = ("Zero" if sc == 0 else
                       "Small" if sc < 0.1 else
                       "Mid" if sc < 1.0 else "Large")
                s(f"initGrad{tag}Norm",
                  "NaN" if row["median_grad_norm"] is None
                  else f"{row['median_grad_norm']:.4f}")
                s(f"initGrad{tag}Finite", f"{row['n_finite']}/{row['n_total']}")
            _z = next((r for r in dg["scales"] if r["branch_init"] == 0), None)
            _l = next((r for r in dg["scales"] if r["branch_init"] == 1.0), None)
            if _z and _l and _z["median_grad_norm"] and _l["median_grad_norm"]:
                s("initGradLargeOverIdentity",
                  sci(_l["median_grad_norm"] / _z["median_grad_norm"]))

    # ---------- Bertini 侧的宏（tab:bertini 正文引用） ----------
    # 数据源：experiments/bertini_decomp.json（不可约分解 + 逐 component 的
    #         loss / 零特征值个数），由 experiments/bertini_decomp.py 驱动
    #         Bertini v1.6 写出。正文绝不手写这些数字。
    bert_path = _exp / "bertini_decomp.json"
    if bert_path.exists():
        bd = {r["label"]: r for r in json.loads(bert_path.read_text(encoding="utf-8"))}

        def _comps(lab, pole=False):
            r = bd.get(lab)
            if not r or r.get("status") != "ok":
                return []
            return [c for c in r["components"]
                    if bool(c.get("pole_only")) == pole]

        allc = [c for lab in bd for c in _comps(lab)]
        s("algBertiniNcomp", str(len(allc)))
        ok = [r for r in bd.values() if r.get("status") == "ok"]
        s("algBertiniMaxTrainParams",
          str(max((r.get("ntrain") or 0) for r in ok)) if ok else "---")
        s("algBertiniMaxSecs",
          "%.0f" % max((r.get("secs") or 0.0) for r in ok) if ok else "---")
        s("algBertiniNsystems", str(len(ok)))
        # 极点伪解（清分母引入）
        npole = sum(1 for lab in bd for c in _comps(lab, pole=True))
        s("algBertiniNpole", str(npole))

        # %H2：所有非极点 component 上 loss 均恒定（spread 在舍入量级）
        spreads = [c.get("loss_spread_rel") for c in allc
                   if c.get("loss_spread_rel") is not None]
        # H2 的真/假不写成 "1"/"0" 宏：正文里会渲染成裸的 "(1)"。
        # 结论改由 analyze.py 在表 caption 里按 JSON 判定后成句。
        # %H3：零特征值个数 == 维数（物理训练参数子空间）

        # plain m=1,n=2：零维分量个数（应与 |I| 一致）、loss 值
        c0 = _comps("plain-depth1-m1-n2")
        s("algBertiniPlainNcomp", str(len(c0)))
        lss = sorted({round(c["loss_mean"][0], 6) for c in c0 if c.get("loss_mean")})
        if "plain-depth1-m1-n2" in bd:
            s("algBertiniPlainLoss",
              "%.4f" % bd["plain-depth1-m1-n2"]["origin"]["loss"][0])
        # plain m=1,n=3：分量 loss（原点值 / 实极小值）
        c1 = _comps("plain-depth1-m1-n3")
        if c1:
            lv = sorted({round(c["loss_mean"][0], 6) for c in c1 if c.get("loss_mean")})
            if lv:
                s("algBertiniNthreeLoss", "%.4f" % lv[-1])
                if len(lv) > 1:
                    s("algBertiniNthreeMinLoss", "%.4f" % lv[0])
        # plain m=2,n=2：正维分量的维数/次数
        c2 = [c for c in _comps("plain-depth1-m2-n2") if int(c["dim"]) > 0]
        if c2:
            s("algBertiniMtwoDim", str(c2[0]["dim"]))
            s("algBertiniMtwoDegree", str(c2[0]["degree"]))
        # cubic 臂：正维分量的 loss（= 100/169 量级）
        cc = [c for c in _comps("cubic-depth1-m1-n2") if int(c["dim"]) > 0]
        if cc and cc[0].get("loss_mean"):
            s("algBertiniCubicDim", str(cc[0]["dim"]))
            s("algBertiniCubicLoss", "%.8f" % cc[0]["loss_mean"][0])
        # 所有真 component 的 witness 点上 |grad L| 的最大值（H2/H3 的数值前提）
        gmax = 0.0
        npts = 0
        for lab, r in bd.items():
            if r.get("status") != "ok":
                continue
            for c in r["components"]:
                if c.get("pole_only"):
                    continue
                for pt in (c.get("points") or []):
                    g = pt.get("grad_norm")
                    if g is None or pt.get("pole") or not pt.get("defined", True):
                        continue
                    gmax = max(gmax, float(g)); npts += 1
        if npts:
            s("algBertiniGradMax", sci(gmax))
        s("algBertiniNprobed", str(npts))
        # H2 的最坏相对散布（真 component）
        spmax = max((c.get("loss_spread_rel") or 0.0)
                    for c in allc) if allc else 0.0
        s("algBertiniSpreadMax", sci(spmax))

        # proj 臂：真分量的维数/次数 + 极点个数 + Bézout
        cp = _comps("proj-m1-n1")
        rp = bd.get("proj-m1-n1")
        if cp:
            s("algBertiniProjDim", str(cp[0]["dim"]))
            s("algBertiniProjDegree", str(cp[0]["degree"]))
            s("algBertiniProjNcomp", str(len(cp)))
        if rp:
            s("algBertiniProjPole", str(len(_comps("proj-m1-n1", pole=True))))
            s("algBertiniProjBezout", str(rp.get("bezout", "---")))
            s("algBertiniProjSecs", "%.1f" % (rp.get("secs") or 0.0))

    # 两样本问题上「最佳直线」的下确界 100/169：与 loss_surface.json 的值比对后
    # 以精确分式形式给出，供正文排版成 \frac{100}{169}。
    if ls_path.exists():
        _inf = ls["d2"]["points"][1]["loss"]
        if abs(_inf - 100.0 / 169.0) < 1e-12:
            s("algInfNum", "100")
            s("algInfDen", "169")

    lines = [
        "% ============================================================",
        "% 由 experiments/make_macros.py 从实验 JSON 自动生成。",
        "% 不要手改：论文中的所有数字都引用这里的宏。",
        "% ============================================================",
    ]
    # 硬校验：LaTeX 控制序列名只能由字母组成。一旦宏名里混进数字或下划线，
    # \newcommand{\L20PolyAcc} 会被 TeX 读成 \L + "20PolyAcc"，报
    # "Missing number" / "Undefined control sequence" 而不是给出可定位的错误。
    # 历史上这个坑复发过三次（\L20Poly*、\synthGauss_highdimBest、\degfiveLam0p001Acc），
    # 因此在写文件前统一拦截，宁可生成失败也不产出坏宏。
    bad = sorted(k for k in M if not k.isalpha())
    if bad:
        raise SystemExit(
            "make_macros: 非法宏名（只能含字母，禁止数字/下划线/点）："
            + ", ".join(bad))
    for k in sorted(M):
        lines.append(r"\newcommand{\%s}{%s}" % (k, M[k]))
    return "\n".join(lines) + "\n", M


# LaTeX 内建/数学命令白名单（用于 --check 时排除，避免误报）
TEX_BUILTINS = set("""
includegraphics input item itemize label left le leq lVert log ln linewidth
maketitle math max min midrule mapsto mathbb mathrm mathcal mathbf nonumber
notin nabla newcommand newline newpage oddsidemargin operatorname odot
overbrace overset overline par paragraph partial usepackage underline upsilon
vspace varphi varepsilon vec vdots verb qquad quad frac forall exists emph
documentclass def displaymath displaystyle dots dot ddots ddot date caption
centering cdot cdots citet citep cite circ clamp binom bigl bigr begin end
bibliography bibliographystyle textbf textit texttt textrm tilde times today
to toprule bottomrule tag textwidth the theta thinspace title tok tt up url
nu xi rho sigma tau phi chi psi omega alpha beta gamma delta epsilon eta zeta
kappa mu pi lambda ll sim simeq sqrt sum prod int inf infty propto pm mp
space special sf sl small subsection subsubsection subcaption subfloat
setcounter section sec ref rVert right rule raggedright approx author bf
tanh hypersetup geometry columnwidth in hat neq eqref newcolumntype
text exp ell geq ge otimessm nc Phi
dim langle rangle noindent
""".split())


def scan_paper_macros(paper_dir):
    used = set()
    pdir = Path(paper_dir)
    files = [pdir / "main.tex"] + sorted(pdir.glob("sections/*.tex"))
    for f in files:
        if f.exists():
            t = f.read_text(encoding="utf-8")
            for m in re.findall(r"\\([A-Za-z]+)", t):
                used.add(m)
    return used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp-dir", default="experiments")
    ap.add_argument("--out", default="paper/macros.tex")
    ap.add_argument("--paper", default="paper")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    recs = load(args.exp_dir)
    text, M = build(recs)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(text, encoding="utf-8")

    missing_num = [k for k, v in M.items() if str(v).startswith("??")]
    print(f"{len(recs)} 条记录 -> {args.out}（{len(M)} 个宏，"
          f"未取到值的宏 {len(missing_num)} 个）")
    if missing_num:
        print("  未取到值的宏:", ", ".join(sorted(missing_num)))

    if args.check:
        used = scan_paper_macros(args.paper)
        undefined = sorted(x for x in used - set(M)
                           if re.match(r"^[a-z]", x) and x not in TEX_BUILTINS)
        print("论文引用但未定义的宏:",
              undefined if undefined else "（无，全部已定义）")


if __name__ == "__main__":
    main()
