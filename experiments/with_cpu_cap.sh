#!/usr/bin/env bash
# 把任意命令限制在不超过 10 个逻辑核内运行。
#
# 背景：本机 20 逻辑核（i7-13650HX，14 物理核）。用户要求所有计算 ≤10 核。
# 两种泄漏途径都要堵：
#   1) 多进程：用 taskset 把 CPU 亲和性钉到 cpu0-9（nproc 会读到 10）；
#   2) 多线程：OpenMP / MKL / OpenBLAS 默认吃满所有核，必须显式设为 10。
#
# 用法：
#   experiments/with_cpu_cap.sh .venv/bin/python experiments/algebraic_critical_points.py
#   experiments/with_cpu_cap.sh sage experiments/algebraic_critical_points.sage
#
# 校验（20 个各 1s CPU 的任务，应约 2s 完成 ≈ 10 核并行）：
#   taskset -c 0-9 nproc        # -> 10
set -euo pipefail

CAP="${CPU_CAP:-10}"
LAST=$((CAP - 1))

export OMP_NUM_THREADS="$CAP"
export MKL_NUM_THREADS="$CAP"
export OPENBLAS_NUM_THREADS="$CAP"
export NUMEXPR_NUM_THREADS="$CAP"
export VECLIB_MAXIMUM_THREADS="$CAP"
# torch 读这个
export OMP_THREAD_LIMIT="$CAP"

echo "[cpu-cap] 亲和性 cpu0-${LAST}（${CAP} 核），线程库上限 ${CAP}" >&2
exec taskset -c "0-${LAST}" "$@"
