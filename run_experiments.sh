#!/usr/bin/env bash
# 统一重跑实验矩阵（干净、单一配置源）。
# 关键点：
#   1) 必须 unset LD_LIBRARY_PATH —— 否则 ~/software 的 C++ libtorch 抢先加载导致 segfault
#   2) 必须 unset *proxy* —— 否则本地子进程继承代理变量
#   3) stdbuf -oL 让日志行缓冲，进度可实时观测（避免 tee/缓冲误导）
#   4) --workers 8：本机 20 逻辑核 / 14 物理核、可用内存 ~9GB，每 worker ~800MB
set -u
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
OUT=experiments/run_all.log

echo "开始: $(date '+%F %T')" | tee "$OUT"
env -u LD_LIBRARY_PATH -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
    -u all_proxy -u ALL_PROXY \
    stdbuf -oL "$PY" experiments/run_all.py --group all --workers 8 2>&1 | tee -a "$OUT"
echo "结束: $(date '+%F %T')" | tee -a "$OUT"
