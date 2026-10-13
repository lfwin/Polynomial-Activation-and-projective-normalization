#!/bin/bash
# 编译论文（自动跑三遍以解析交叉引用与 bibtex）。
#
# 用法: ./make_paper.sh
#
# 依赖: pdflatex / bibtex（TeX Live）。论文正文只引用 macros.tex 里的宏，
# 表格由 experiments/analyze.py 生成，图片为 PDF。
set -euo pipefail

cd "$(dirname "$0")"
PAPER=paper

if [ ! -f "$PAPER/macros.tex" ]; then
  echo "[warn] $PAPER/macros.tex 不存在，先运行:"
  echo "       python3 experiments/make_macros.py"
fi

cd "$PAPER"

echo "== pass 1/3 =="
pdflatex -interaction=nonstopmode -halt-on-error main.tex > /tmp/tex1.log 2>&1 || {
  echo "!! pass 1 失败，错误摘要："; grep -n -A3 "^!" /tmp/tex1.log | head -60; exit 1; }

echo "== bibtex =="
bibtex main > /tmp/texb.log 2>&1 || echo "   (bibtex 警告见 /tmp/texb.log)"

echo "== pass 2/3 =="
pdflatex -interaction=nonstopmode -halt-on-error main.tex > /tmp/tex2.log 2>&1 || {
  echo "!! pass 2 失败："; grep -n -A3 "^!" /tmp/tex2.log | head -60; exit 1; }

echo "== pass 3/3 =="
pdflatex -interaction=nonstopmode -halt-on-error main.tex > /tmp/tex3.log 2>&1 || {
  echo "!! pass 3 失败："; grep -n -A3 "^!" /tmp/tex3.log | head -60; exit 1; }

echo
echo "== 未解析引用 / 警告 =="
grep -c "Warning" /tmp/tex3.log || true
grep -n "undefined" /tmp/tex3.log | head -20 || true

if [ -f main.pdf ]; then
  echo
  echo "OK  -> $(pwd)/main.pdf"
  pdfinfo main.pdf 2>/dev/null | grep -E "^(Pages|File size)" || true
else
  echo "!! 未生成 main.pdf"; exit 1
fi
