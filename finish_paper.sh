#!/usr/bin/env bash
# 一步一步把论文跑完：
#   1) 等当前 run_all.py 批次结束
#   2) 再跑一次 --group all（resume，跳过已有 JSON），补齐后来新增的配置
#      —— 例如 tanh（加进 ACT_LIST 时本轮作业列表已枚举完）与 ref_orig（20 epochs）
#   3) 生成表格/图（analyze.py）
#   4) 生成 macros.tex 并校验宏（make_macros.py --check）
#   5) 编译论文（make_paper.sh）
#
# 用法: ./finish_paper.sh
set -u
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
LOG=experiments/finish.log
ENVV="env -u LD_LIBRARY_PATH -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY"

say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOG"; }

# ---- 1) 等当前批次结束 ----
# 注意：run_all.py 的 resume 起点与这里的等待点之间没有原子性。若父进程在
# 子进程尚未入表时就退出，pgrep 会误判"已结束"从而跳过补齐。因此要求连续两次
# 检查都为空才认为真的结束。
say "等待当前 run_all.py 批次结束（每 60s 检查一次）..."
idle=0
while true; do
  if pgrep -f "run_all.py --group" > /dev/null 2>&1; then
    idle=0
  else
    idle=$((idle + 1))
    [ "$idle" -ge 2 ] && break
  fi
  sleep 60
done
say "当前批次已结束（连续两次检查均无 run_all.py）。"

# ---- 2) 补齐新增配置 ----
say "补齐新增配置（resume）..."
$ENVV stdbuf -oL "$PY" experiments/run_all.py --group all --workers 8 2>&1 | tee -a "$LOG"
say "配置补齐完成。"

# ---- 3) 表/图 ----
say "生成表格与图..."
$ENVV "$PY" experiments/analyze.py 2>&1 | tee -a "$LOG"

# ---- 4) 宏 ----
say "生成 macros.tex 并校验..."
$ENVV "$PY" experiments/make_macros.py --check 2>&1 | tee -a "$LOG"

# ---- 5) 编译 ----
say "编译论文..."
bash ./make_paper.sh 2>&1 | tee -a "$LOG"

say "全部完成。"
