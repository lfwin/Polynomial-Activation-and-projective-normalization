#!/usr/bin/env python3
"""
保真度校验：重跑若干代表性配置，与已存 JSON 比对。

用途：代码在实验过程中改过（新增激活、修 diagnostics、门控宽度对齐、
gate_act 参数）。若某组配置走的是未改动的代码路径，重跑结果应与已存
JSON 逐位一致 —— 那样就能保留已存结果，不必全量重跑。

用法: python experiments/verify_fidelity.py
"""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "experiments"))
import run_all  # noqa: E402

# MNIST 原始文件不入库；用 RESULT_DATA_DIR 指到你放 MNIST 的目录。
DATA_DIR = os.environ.get("MNIST_DIR", str(ROOT / "data"))
VERIFY_DIR = ROOT / "experiments" / "_verify"
FIELDS = ["test_accuracy", "train_accuracy", "n_params"]

# 每个大组挑一个代表性配置
PICKS = [
    ("act_mnist",    dict(dataset="mnist", activation="relu", norm="none", depth=10, hidden=128, seed=42)),
    ("norm_mnist",   dict(dataset="mnist", activation="poly", norm="l2", depth=10, hidden=128, seed=42)),
    ("norm_mnist",   dict(dataset="mnist", activation="poly", norm="batch", depth=10, hidden=128, seed=42)),
    ("norm_mnist",   dict(dataset="mnist", activation="relu", norm="layer", depth=10, hidden=128, seed=42)),
    ("seeds",        dict(dataset="mnist", activation="poly", norm="l2", depth=10, hidden=128, seed=0)),
    ("depth_mnist",  dict(dataset="mnist", activation="poly", norm="l2", depth=20, hidden=128, seed=42)),
    ("degree_mnist", dict(dataset="mnist", activation="poly", degree=4, lam=0.01, norm="l2", depth=10, hidden=128, seed=42)),
    ("synth3",       dict(dataset="synth_train", synth_name="rings", activation="poly", norm="l2", depth=3, hidden=64)),
]


def find_stored(group, q):
    d = ROOT / "experiments" / group
    hits = []
    for f in d.glob("*.json"):
        try:
            r = json.load(open(f))
        except Exception:
            continue
        a = r["args"]
        if all(a.get(k) == v for k, v in q.items()):
            hits.append((f, r))
    return hits


def main():
    g = run_all.build_groups()
    # 建 (group, args-tuple) -> cfg 索引
    byargs = {}
    for name, cfgs in g.items():
        for c in cfgs:
            byargs.setdefault(name, []).append(c)

    VERIFY_DIR.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env.pop("LD_LIBRARY_PATH", None)

    ok = bad = skip = 0
    for group, q in PICKS:
        stored = find_stored(group, q)
        if not stored:
            print(f"[skip] {group} {q} 无已存 JSON")
            skip += 1
            continue
        # 找对应 cfg
        cfg = None
        for c in byargs.get(group, []):
            if all(str(c.get(k)) == str(v) for k, v in q.items() if k in c):
                if c.get("out") == group or True:
                    # 需要 args 全部匹配
                    if all(c.get(k, None) == v for k, v in q.items()):
                        cfg = c
                        break
        if cfg is None:
            print(f"[skip] {group} {q} 未在矩阵中找到配置")
            skip += 1
            continue

        cmd = run_all.cfg_to_cmd(cfg)
        # 把 --out-dir 换成校验目录，并显式指向数据目录
        i = cmd.index("--out-dir")
        cmd[i + 1] = str(VERIFY_DIR)
        cmd += ["--data-dir", DATA_DIR]
        r = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(ROOT))
        tag = cfg["tag"]
        new_files = list(VERIFY_DIR.glob(f"*{tag}*.json"))
        if r.returncode != 0 or not new_files:
            print(f"[FAIL] {tag} 重跑失败 rc={r.returncode}")
            print("   ", (r.stderr or r.stdout)[-300:])
            bad += 1
            continue
        new = json.load(open(new_files[0]))
        old = stored[0][1]
        diffs = []
        for f in FIELDS:
            a, b = old.get(f), new.get(f)
            if a is None or b is None:
                continue
            if isinstance(a, float):
                if abs(a - b) > 1e-9:
                    diffs.append(f"{f}: {a} -> {b}")
            elif a != b:
                diffs.append(f"{f}: {a} -> {b}")
        # 历史曲线逐位比对
        for key in ("test_acc", "train_acc"):
            ha, hb = old["history"].get(key), new["history"].get(key)
            if ha and hb:
                md = max(abs(x - y) for x, y in zip(ha, hb))
                if md > 1e-9:
                    diffs.append(f"history.{key} maxdiff={md:.3e}")

        if diffs:
            print(f"[DIFF] {tag}")
            for d in diffs:
                print(f"        {d}")
            bad += 1
        else:
            print(f"[same] {tag}  ({old['test_accuracy']*100:.2f}% 逐位一致)")
            ok += 1

    print(f"\n结论: 一致 {ok}，不一致 {bad}，跳过 {skip}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
