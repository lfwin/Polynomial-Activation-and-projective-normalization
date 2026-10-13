#!/usr/bin/env bash
# 实验进度速览
cd "$(dirname "$0")/.."
total=$(grep -c 'OK \|ERR(' experiments/run_all.log 2>/dev/null || echo 0)
err=$(grep -c 'ERR(' experiments/run_all.log 2>/dev/null || echo 0)
echo "进度: $total / 297    失败: $err"
echo "--- 最近 3 行 ---"
tail -3 experiments/run_all.log
echo "--- 负载 / 内存 ---"
uptime
free -m | head -2
echo "--- 运行中的 worker 数 ---"
pgrep -cf "harness.py" || echo 0
