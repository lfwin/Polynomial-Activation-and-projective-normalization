#!/usr/bin/env python3
"""
实验矩阵 + 并行执行器。

用法:
  python experiments/run_all.py --list                 # 只列出所有配置
  python experiments/run_all.py --group act_mnist      # 跑某一组
  python experiments/run_all.py --group all            # 跑全部
  python experiments/run_all.py --group all --workers 12

每个配置一个子进程，OMP_NUM_THREADS=1（实测多线程几乎无收益，单线程多进程最快）。
已存在的 JSON 会跳过（断点续跑）；加 --force 重跑。
"""
import argparse
import os
import pathlib
import subprocess
import sys
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
# 用当前解释器运行 worker：谁调用 run_all.py，就用谁跑 harness.py。
# 这样无论 venv 建在哪里、叫什么都成立（可用 PYTHON 环境变量覆盖）。
PY = os.environ.get("PYTHON", sys.executable)
HARNESS = str(HERE / "harness.py")

# 常用配置
C = dict(hidden=128, depth=10, lr=1e-3, epochs=10, batch_size=128, seed=42)


def base(**kw):
    cfg = dict(C)
    cfg.update(kw)
    return cfg


# ---------------------------------------------------------------
# 组 1：MNIST 上用不同激活函数做纯消融（无归一化）
# 只保留两类：
#   (a) 当前 LLM 真正在用的激活：relu / gelu / silu，以及门控 swiglu / geglu
#   (b) 论文主角：多项式族（poly / poly_proj / legendre / chebyshev / bspline）
# identity = 线性对照（不是激活，用于说明非线性本身的作用）
# ---------------------------------------------------------------
ACT_LIST = [
    "identity",        # 线性对照（说明非线性本身的作用）
    "tanh", "relu",    # 经典基线
    "gelu", "silu",    # 现代 LLM 常用逐元素激活
    "poly",            # z + 0.01 z^3，无归一化
    "poly_proj",       # 论文原始：多项式 + 射影 L2 归一化
]

# 门控族：只用真实模型在用的 SwiGLU / GeGLU，外加多项式门控 polyglu
# （直接服务论文的"多项式"主题）。门控有两个线性层，宽度取 82 即可与
# 128 宽的逐元素网络参数量对齐：238466 ≈ 233866（差 2%）。
GATED_LIST = ["swiglu", "geglu", "polyglu"]
GATED_H = 82

# 可学习多项式激活族（"不同多项式激活函数"的学习版）
LEARN_POLY_LIST = ["legendre", "chebyshev", "bspline"]

NORM_LIST = [
    ("none", "post"), ("batch", "post"), ("layer", "post"), ("rms", "post"),
    ("layer", "pre"), ("rms", "pre"),
]


def build_groups():
    G = {}

    # ---- ref_orig：严格复现原脚本 train_projective_poly_mnist_10layers.py ----
    # 原脚本 argparse 默认 epochs=20（不是 10），其余（AdamW, lr1e-3,
    # weight_decay1e-4, CosineAnnealingLR T_max=epochs, clip 5.0,
    # xavier_uniform + zero bias, batch 128, seed 42, poly_proj 无归一化）
    # 与 harness 完全一致。
    # legacy_loader + no_diagnostics 是「逐位复现」的充要条件（实测）：
    #   两者同开 -> 20ep 得 96.86%，与原脚本 results.json 的 0.9686 完全一致；
    #   缺任一 -> 洗牌顺序被显式 generator 或诊断前向改变，得 96.95%（差 0.09）。
    G["ref_orig"] = [base(activation="poly_proj", norm="none", dataset="mnist",
                          epochs=20, out="ref_orig", tag="mnist-L10-ref-orig",
                          legacy_loader=True, no_diagnostics=True)]

    # ---- ref_fidelity：2-epoch 保真度校验（与 verify_fidelity.py 同配置）----
    # 同一网络同一超参数，只是 epochs=2：因为余弦调度周期 = epochs，
    # 只有同样按 2 epochs 跑，调度才与原脚本一致，才能逐位比对。
    G["ref_fidelity"] = [base(activation="poly_proj", norm="none",
                              dataset="mnist", epochs=2,
                              out="ref_fidelity", tag="mnist-L10-fid",
                              legacy_loader=True, no_diagnostics=True)]

    g = []
    for a in ACT_LIST:
        g.append(base(activation=a, norm="none", dataset="mnist",
                      out="act_mnist", tag="mnist-L10"))
    for a in GATED_LIST:
        g.append(base(activation=a, norm="none", dataset="mnist",
                      hidden=GATED_H, out="act_mnist", tag="mnist-L10"))
    G["act_mnist"] = g

    # ---- glu_norm_mnist：门控单元 x 归一化（GLU 的真实用法；无归一化会数值崩溃）----
    g = []
    for a in GATED_LIST:
        for norm in ["none", "l2", "batch", "layer", "rms"]:
            g.append(base(activation=a, norm=norm, dataset="mnist",
                          hidden=GATED_H, out="glu_norm_mnist",
                          tag=f"mnist-L10-{a}-{norm}"))
    # 逐元素对照（保持 H=128 与门控 H=82 参数量对齐）
    for a in ["relu", "gelu", "silu"]:
        for norm in ["none", "layer", "batch", "rms", "l2"]:
            g.append(base(activation=a, norm=norm, dataset="mnist",
                          out="glu_norm_mnist", tag=f"mnist-L10-{a}-{norm}"))
    G["glu_norm_mnist"] = _dedupe(g)

    # ---- learn_poly_mnist：可学习多项式激活（Legendre / Chebyshev / B-spline）----
    g = []
    for a in LEARN_POLY_LIST:
        for norm in ["none", "l2"]:
            g.append(base(activation=a, norm=norm, dataset="mnist",
                          out="learn_poly_mnist", tag="mnist-L10-learnpoly"))
        g.append(base(activation=a, norm="batch", dataset="mnist",
                      out="learn_poly_mnist", tag="mnist-L10-learnpoly"))
    # 阶数消融（Legendre）
    for K in [3, 6, 10, 16]:
        g.append(base(activation="legendre", lpoly_degree=K, norm="none",
                      dataset="mnist", out="learn_poly_mnist",
                      tag=f"mnist-L10-legendre-K{K}"))
    # B-spline 网格数消融
    for G_ in [4, 8, 16]:
        g.append(base(activation="bspline", bspline_grid=G_, norm="none",
                      dataset="mnist", out="learn_poly_mnist",
                      tag=f"mnist-L10-bspline-G{G_}"))
    G["learn_poly_mnist"] = g

    # ---- width_mnist：宽度对齐消融（128 vs 82，逐元素激活）----
    g = []
    for w in [82, 128, 192]:
        for act, norm in [("relu", "none"), ("gelu", "none"),
                          ("poly_proj", "none"), ("poly", "l2")]:
            g.append(base(activation=act, norm=norm, hidden=w, dataset="mnist",
                          out="width_mnist", tag=f"mnist-W{w}-{act}"))
    G["width_mnist"] = g

    # ---- norm_mnist：归一化方式对比（固定 ReLU 与 poly，各一组）----
    g = []
    for act in ["relu", "poly"]:
        for norm, pos in NORM_LIST:
            g.append(base(activation=act, norm=norm, norm_pos=pos,
                          dataset="mnist", out="norm_mnist",
                          tag=f"mnist-L10-{act}"))
    # ReLU + 射影 l2：论文用它论证"射影归一化不可跨激活函数直接套用"
    # （ReLU 输出非负、聚在远离原点的锥里，强行投影掉幅度线索反而更差）。
    # 这个配置必须跑在 norm_mnist 组内，否则论文只能去 glu_norm_mnist
    # 的对照记录里借值，跨组取数正是本项目反复出现的幻值来源。
    g.append(base(activation="relu", norm="l2", norm_pos="post",
                  dataset="mnist", out="norm_mnist", tag="mnist-L10-relu-l2"))
    # poly + 每种归一化（含 l2 = 射影）
    for norm, pos in NORM_LIST + [("l2", "post")]:
        if norm == "none" and pos == "post":
            pass
        g.append(base(activation="poly", norm=norm, norm_pos=pos,
                      dataset="mnist", out="norm_mnist",
                      tag="mnist-L10-poly-all"))
    G["norm_mnist"] = _dedupe(g)

    # ---- degree_mnist：多项式阶数 x lam 网格 ----
    g = []
    for deg in [2, 3, 4, 5]:
        for lam in [0.001, 0.01, 0.05, 0.1]:
            g.append(base(activation="poly", degree=deg, lam=lam, norm="l2",
                          dataset="mnist", out="degree_mnist",
                          tag=f"mnist-L10-deg{deg}-lam{lam:g}"))
    G["degree_mnist"] = g

    # ---- depth_mnist：深度扩展性（2/3/5/10/15/20 层）----
    g = []
    for d in [2, 3, 5, 10, 15, 20]:
        for act, norm, tag in [("relu", "none", "relu"),
                               ("gelu", "none", "gelu"),
                               ("identity", "none", "identity"),
                               ("poly", "l2", "polyproj"),
                               ("poly", "batch", "poly-bn"),
                               ("poly", "layer", "poly-ln"),
                               ("poly", "none", "poly")]:
            g.append(base(activation=act, norm=norm, depth=d,
                          dataset="mnist", out="depth_mnist",
                          tag=f"mnist-L{d}-{tag}"))
    G["depth_mnist"] = g

    # ---- synth3：自构数据 + 3 层小网络（用户指定）----
    g = []
    synth = [("rings", 2), ("spiral", 2), ("blobs", 2), ("xor", 2),
             ("checker", 2), ("cubes", 3), ("parity", 10),
             ("gauss_highdim", 50)]
    synth_acts = ["identity", "relu", "gelu", "silu", "tanh",
                  "poly", "swiglu", "geglu"]
    for name, indim in synth:
        for a in synth_acts:
            norm = "l2" if a == "poly" else "none"
            cfg = dict(hidden=64, depth=3, lr=3e-3, epochs=60,
                       batch_size=64, seed=0)
            cfg.update(activation=a, norm=norm, dataset="synth_train",
                       synth_name=name, synth_n=4000, synth_seed=7,
                       in_dim=indim, out="synth3",
                       tag=f"synth-{name}-{a}")
            g.append(cfg)
    G["synth3"] = g

    # ---- synth3_norm：自构数据上的归一化对比 ----
    g = []
    for name, indim in [("rings", 2), ("spiral", 2), ("checker", 2),
                        ("parity", 10)]:
        for act, norm, pos in [("poly", "l2", "post"), ("poly", "none", "post"),
                               ("poly", "batch", "post"), ("poly", "layer", "post"),
                               ("poly", "rms", "post"), ("relu", "batch", "post"),
                               ("relu", "layer", "post"), ("relu", "none", "post")]:
            cfg = dict(hidden=64, depth=3, lr=3e-3, epochs=60, batch_size=64,
                       seed=0)
            cfg.update(activation=act, norm=norm, norm_pos=pos,
                       dataset="synth_train", synth_name=name, synth_n=4000,
                       synth_seed=7, in_dim=indim, out="synth3_norm",
                       tag=f"synthn-{name}-{act}-{norm}")
            g.append(cfg)
    G["synth3_norm"] = g

    # ---- synth3_depth：自构数据（parity）深度扩展性 ----
    g = []
    for d in [3, 5, 10]:
        for act, norm in [("relu", "none"), ("poly", "l2"), ("poly", "none"),
                          ("poly", "batch"), ("poly", "layer"), ("swiglu", "none")]:
            cfg = dict(hidden=64, depth=d, lr=3e-3, epochs=60, batch_size=64,
                       seed=0)
            cfg.update(activation=act, norm=norm, dataset="synth_train",
                       synth_name="parity", synth_n=4000, synth_seed=7,
                       in_dim=10, out="synth3_depth",
                       tag=f"synthd-parity-L{d}-{act}-{norm}")
            g.append(cfg)
    G["synth3_depth"] = g

    # ---- seeds：重复性（MNIST，5 个种子）----
    g = []
    for s in [0, 1, 2, 3, 4]:
        for act, norm, tag in [("relu", "none", "relu"), ("gelu", "none", "gelu"),
                               ("poly", "l2", "polyproj"),
                               ("poly", "batch", "poly-bn"),
                               ("poly", "layer", "poly-ln")]:
            cfg = base(activation=act, norm=norm, seed=s, dataset="mnist",
                       out="seeds", tag=f"seed-{act}-{norm}")
            g.append(cfg)
    G["seeds"] = g

    # ---- depth_very：极深（30/50 层），检验梯度传播 ----
    g = []
    for d in [30, 50]:
        for act, norm, tag in [("relu", "none", "relu"),
                               ("relu", "batch", "relu-bn"),
                               ("relu", "layer", "relu-ln"),
                               ("identity", "none", "identity"),
                               ("poly", "l2", "polyproj"),
                               ("poly", "layer", "poly-ln"),
                               ("poly", "none", "poly")]:
            cfg = base(activation=act, norm=norm, depth=d, dataset="mnist",
                       out="depth_very", tag=f"mnist-L{d}-{tag}")
            g.append(cfg)
    G["depth_very"] = g

    # ---- residual_mnist：多项式激活 + 恒等残差，**不含任何归一化** ----
    # 主线：论文里无归一化的 poly 在浅层尚可、到 L30 即塌成常数预测器。
    # 这里问一个独立的问题：只加恒等残差（不加任何 normalization），
    # 极深网络能不能重新训起来？答案是能，且 100--300 层基本平。
    #
    # 两种放置方式（首层 784->128 维度不匹配，不做 skip）：
    #   none  h^i = f(W^i h^{i-1})                纯多项式
    #   r1    h^i = f(W^i h^{i-1}) + h^{i-1}      恒等残差，求和后不再归一化
    # 两种初始化：
    #   xavier 论文默认（W 元素量级 1/sqrt(dim)，残差是"可训练扰动"）
    #   id     截断单位阵 + 0.1 倍 xavier 扰动（让每层接近恒等映射）
    #
    # 该组单独放在 residual_mnist/ 目录，并在 analyze/make_macros 里被
    # _SOLO_GROUPS 排除，避免 L100 的记录污染主网格的按深度查询。
    g = []
    for mode in ["none", "r1"]:
        for initm in ["xavier", "id"]:
            for s in [42, 43, 44, 45]:
                cfg = base(activation="poly", norm="none", depth=100,
                           dataset="mnist", out="residual_mnist",
                           residual=mode, init_mode=initm, seed=s,
                           tag=f"mnist-L100-poly-res{mode}-{initm}")
                g.append(cfg)
    # 深度扫描（xavier，2 种子）：看残差把"可训练深度"推到多远
    for d in [10, 50, 200, 300]:
        for mode in ["none", "r1"]:
            for s in [42, 43]:
                cfg = base(activation="poly", norm="none", depth=d,
                           dataset="mnist", out="residual_mnist",
                           residual=mode, init_mode="xavier", seed=s,
                           tag=f"mnist-L{d}-poly-resdepth")
                g.append(cfg)
    # 现代默认激活的对照：ReLU + 残差，同样无归一化
    for s in [42, 43]:
        cfg = base(activation="relu", norm="none", depth=100,
                   dataset="mnist", out="residual_mnist",
                   residual="r1", init_mode="xavier", seed=s,
                   tag="mnist-L100-relu-resr1")
        g.append(cfg)
    G["residual_mnist"] = g

    # ---- pure_poly_mnist：纯多项式，无 bias、无 clamp、无残差 ----
    # 用户问的问题：只用多项式激活（z + 0.01 z^3），去掉 bias 与截断，
    # 也不加残差连接，能不能训 100 层？
    # 答案：不能，而且失败有两种不同机制，取决于有没有射影归一化。
    #   无归一化：层范数几何增长 -> 第 1 个 epoch 内数值溢出（train_loss 变 nan）。
    #   射影归一化（norm=l2）：每层范数被钉死在 1.0，数值稳定，但层间梯度
    #     衰减到 ~1e-12，L30 还行（90%），L50 掉到 65%，L100 塌成常数（11%）。
    # 结论：scale 控制是"能不能加深"的决定因素，而射影归一化只把可训练
    # 深度从 ~15 推到 ~30，推不到 100；残差才是到 100+ 的必要条件。
    g = []
    for d in [10, 20, 30, 50, 100]:
        for norm in ["none", "l2"]:
            cfg = base(activation="poly", norm=norm, depth=d, dataset="mnist",
                       out="pure_poly_mnist", clamp="none", no_bias=True,
                       residual="none", init_mode="xavier", seed=42,
                       tag=f"mnist-L{d}-purepoly-{norm}")
            g.append(cfg)
    # pre-norm 放置对照：归一化放在分支内（恒等路径不受影响）是否能救 L100
    for d in [100]:
        for pos in ["pre", "post"]:
            cfg = base(activation="poly", norm="l2", norm_pos=pos, depth=d,
                       dataset="mnist", out="pure_poly_mnist", clamp="none",
                       no_bias=True, residual="none", init_mode="xavier",
                       seed=42, tag=f"mnist-L{d}-purepoly-l2-pos{pos}")
            g.append(cfg)
    # 初始化轴：对角初始化 W = diag(1 + 0.1 U(-1,1))，非对角严格为 0
    # （不做整矩阵 xavier 填充）。效果：可训练深度从 ~30 推到 ~50
    # （L50 11.4% -> 90.1%），但 L100 仍塌（~10%），层间梯度降到 ~1e-10
    # —— 初始化能推迟崩溃，不能消除它。
    for d in [10, 30, 50, 100]:
        cfg = base(activation="poly", norm="l2", depth=d, dataset="mnist",
                   out="pure_poly_mnist", clamp="none", no_bias=True,
                   residual="none", init_mode="diag", seed=42,
                   tag=f"mnist-L{d}-purepoly-l2-diag")
        g.append(cfg)
    # 初始化轴 2（diag0）：首层 784->128 无法做成真对角阵，若强行"对角"只保留
    # 前 128 个输入像素，MNIST 顶部约 2/3 的图片那里全黑 -> 隐层恒为 0 ->
    # L2 反向除以零范数 -> 第 1 步 nan（见 init="diag"）。这里对非方阵层改用
    # 随机初始化、方阵隐层保持 W = diag(1 + 0.1 U(-1,1))。
    # 效果：L10/30/50 = 93.06/92.78/91.87%（对比纯 diag 全 9.80%、xavier 在
    # L50 已崩到 11.35%），把可训练深度推到 ~50；L100 仍塌（13.31%），
    # 层间梯度 ~6e-10。两种种子一致（L50 91.87/92.32，L100 13.31/11.95）。
    for d in [10, 30, 50, 100]:
        cfg = base(activation="poly", norm="l2", depth=d, dataset="mnist",
                   out="pure_poly_mnist", clamp="none", no_bias=True,
                   residual="none", init_mode="diag0", seed=42,
                   tag=f"mnist-L{d}-purepoly-l2-diag0")
        g.append(cfg)
    # 可学习三次激活 a + b z + c z^2 + d z^3（a,b,c,d 逐通道可训练），
    # 无 bias、无 clamp；系数初始 a=0,b=1,c=0,d=0.01（= 固定版 z+0.01z^3 的起点）。
    # 2x2：有无 projective norm(l2) x 有无恒等残差(r1)，深度 50 与 100。
    # 结论：projective norm 与残差必须同时在场，缺一不可。
    #   l2 + r1   : L50 96.9%  L100 97.60%   <- 唯一能到 100 层的组合
    #   l2 无残差 : L50 96.35% L100 11.35%
    #   r1 无归一化: L50/L100 均 9.80%（前向第 12 层溢出）
    # 可学习激活比固定激活略好（97.60% vs 97.5%），但让残差路径更脆弱：
    # 系数离开初始点后等效 Lipschitz 常数变大，无归一化时几何级数直接爆掉。
    for d in [50, 100]:
        for norm, res in [("l2", "none"), ("l2", "r1"),
                          ("none", "r1"), ("none", "none")]:
            cfg = base(activation="poly_learn", lam=0.01, norm=norm, depth=d,
                       dataset="mnist", out="pure_poly_mnist", clamp="none",
                       no_bias=True, residual=res, init_mode="xavier",
                       coeff_init="identity", seed=42,
                       tag=f"mnist-L{d}-polylearn-{norm}-{res}")
            g.append(cfg)
    G["pure_poly_mnist"] = g

    return G


def _dedupe(configs):
    seen, out = set(), []
    for c in configs:
        key = json.dumps(c, sort_keys=True)
        if key not in seen:
            seen.add(key)
            out.append(c)
    return out


def _expected_file(cfg):
    """复刻 harness.py 的 cid 命名，得到该配置唯一对应的 JSON 文件名（不含 .json）。

    旧逻辑用 glob("*{tag}*.json") 判重，但多个配置会共用同一个 tag
    （act_mnist 的 10 个配置 tag 全是 "mnist-L10"，norm_mnist 的 6 个配置
    tag 全是 "mnist-L10-poly"）。glob 只要命中该组任一文件就跳过整组，
    于是后加入 ACT_LIST 的 tanh 被永久漏跑、ref_orig/ref_fidelity 也补不上。
    改为按完整文件名精确匹配。
    """
    tag = f"{cfg['tag']}_" if cfg.get("tag") else ""
    norm_pos = "-pre" if cfg.get("norm_pos", "post") == "pre" else ""
    # 残差/初始化/纯多项式轴（clamp、bias）必须逐字复刻 harness.py 的 cid 拼接，
    # 否则 runner 认不出自己刚写出的结果，会把同一批配置无限重跑。
    # harness.py 的顺序：-{residual} -> -{init} -> -nobias -> -noclamp。
    res_sfx = ""
    if cfg.get("residual", "none") != "none":
        res_sfx += f"-{cfg['residual']}"
    if cfg.get("init_mode", "xavier") != "xavier":
        res_sfx += f"-{cfg['init_mode']}"
    if cfg.get("no_bias"):
        res_sfx += "-nobias"
    clamp = cfg.get("clamp", 10.0)
    if clamp is None or str(clamp).lower() in ("none", "null", ""):
        res_sfx += "-noclamp"
    elif float(clamp) != 10.0:
        res_sfx += f"-clamp{float(clamp):g}"
    if cfg.get("coeff_init", "identity") != "identity":
        res_sfx += f"-coeff{cfg['coeff_init']}"
    return (
        f"{tag}{cfg.get('dataset', 'mnist')}-{cfg.get('synth_name', 'rings')}-"
        f"{cfg.get('activation', 'relu')}-d{cfg.get('degree', 3)}-"
        f"lam{cfg.get('lam', 0.01):g}-{cfg.get('norm', 'none')}{norm_pos}"
        f"{res_sfx}-"
        f"L{cfg.get('depth', 10)}-H{cfg.get('hidden', 128)}-"
        f"lr{cfg.get('lr', 1e-3):g}-e{cfg.get('epochs', 10)}-s{cfg.get('seed', 42)}"
    )


def cfg_to_cmd(cfg):
    """把配置字典转成命令行参数列表。"""
    out = cfg.get("out", "runs")
    cmd = [PY, HARNESS]
    mapping = {
        "activation": "--activation", "degree": "--degree", "lam": "--lam",
        "norm": "--norm", "norm_pos": "--norm-pos", "depth": "--depth",
        "hidden": "--hidden", "epochs": "--epochs", "batch_size": "--batch-size",
        "lr": "--lr", "seed": "--seed", "dataset": "--dataset",
        "synth_name": "--synth-name", "synth_n": "--synth-n",
        "synth_seed": "--synth-seed", "in_dim": "--in-dim", "tag": "--tag",
        "lpoly_degree": "--lpoly-degree", "bspline_grid": "--bspline-grid",
        "residual": "--residual", "init_mode": "--init",
        "clamp": "--clamp", "no_bias": "--no-bias",
        "coeff_init": "--coeff-init",
    }
    for k, flag in mapping.items():
        if k in cfg and cfg[k] is not None:
            if isinstance(cfg[k], bool):
                if cfg[k]:
                    cmd += [flag]
                continue
            cmd += [flag, str(cfg[k])]
    # 布尔开关：参考组必须开这两个才能真正逐位复现原脚本。
    # legacy_loader: 用 DataLoader 的全局 RNG shuffle（原脚本做法），而非显式
    #   torch.Generator —— 原脚本没有 generator 参数，两者洗牌顺序不同。
    # no_diagnostics: 关掉诊断前向。诊断会在 legacy 模式下消耗/推进全局 RNG，
    #   把 shuffle 顺序整体挪位（实测 epoch1 87.00% -> 87.15%）。开 legacy
    #   却留诊断，等于既没有复现又不自洽；参考组必须两者同时开。
    if cfg.get("legacy_loader"):
        cmd += ["--legacy-loader"]
    if cfg.get("no_diagnostics"):
        cmd += ["--no-diagnostics"]
    # 记录相对路径（run_one 以 ROOT 为 cwd，相对路径同样可用）：
    # 仓库约定不得把本机绝对路径写进配置/结果 JSON。
    cmd += ["--out-dir", str(pathlib.PurePath("experiments") / out)]
    return cmd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--group", default="all")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    G = build_groups()
    if args.list:
        total = 0
        for name, cfgs in G.items():
            print(f"{name:16s} {len(cfgs):4d} 个配置")
            total += len(cfgs)
        print(f"{'TOTAL':16s} {total:4d}")
        return

    groups = list(G) if args.group == "all" else [args.group]

    jobs = []
    for name in groups:
        out_dir = ROOT / "experiments" / G[name][0]["out"]
        out_dir.mkdir(parents=True, exist_ok=True)
        for cfg in G[name]:
            cmd = cfg_to_cmd(cfg)
            cid = cfg["tag"]
            if not args.force and (out_dir / f"{_expected_file(cfg)}.json").exists():
                continue
            jobs.append((name, cid, cmd, out_dir))

    print(f"待运行: {len(jobs)} 个配置，并行度 {args.workers}", flush=True)
    if not jobs:
        return

    env_base = dict(os.environ)
    env_base["OMP_NUM_THREADS"] = "1"
    env_base["MKL_NUM_THREADS"] = "1"
    env_base["OPENBLAS_NUM_THREADS"] = "1"
    env_base["NUMEXPR_NUM_THREADS"] = "1"
    env_base.pop("LD_LIBRARY_PATH", None)   # 关键：避免 libtorch 冲突

    log_dir = ROOT / "experiments" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    def run_one(job):
        name, cid, cmd, out_dir = job
        log = log_dir / f"{cid}.log"
        with open(log, "w") as f:
            r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT,
                               env=env_base, cwd=str(ROOT))
        # 取最后一次 epoch 行
        tail = ""
        try:
            lines = [l for l in log.read_text().splitlines() if l.strip()]
            tail = lines[-1] if lines else ""
        except Exception:
            tail = ""
        return name, cid, r.returncode, tail

    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(run_one, j): j for j in jobs}
        for fut in as_completed(futs):
            name, cid, rc, tail = fut.result()
            done += 1
            flag = "OK " if rc == 0 else f"ERR({rc})"
            print(f"[{done}/{len(jobs)}] {flag} {cid} :: {tail}", flush=True)


if __name__ == "__main__":
    main()
