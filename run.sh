#!/usr/bin/env bash
# 运行 10 层射影多项式网络训练。
#
# 关键：必须清掉全局 LD_LIBRARY_PATH。
# 若你的 shell 里 LD_LIBRARY_PATH 指向某个 C++ libtorch 安装目录，它会抢先
# 加载自己的 libtorch_cpu.so / libc10.so，与 venv 里的 torch 符号冲突，
# import torch 直接 segfault。
set -euo pipefail

cd "$(dirname "$0")"
unset LD_LIBRARY_PATH

PY="${PYTHON:-python3}"
exec "$PY" train_projective_poly_mnist_10layers.py "$@"
