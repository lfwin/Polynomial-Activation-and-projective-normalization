# arXiv 单文件构建

把论文打成**一个自包含的 `main.tex`**：零子目录、零外部图片、零 bibtex 依赖。

## 为什么需要这个

arXiv 的编译环境反复只收到 `main.tex` —— 子目录里的文件（第一次 `sections/`、第二次 `refs.bib`、第三次 `figures/`）全部丢失。图缺失时 pdflatex 仍能产出 PDF，但**退出码是 1**，arXiv 据此判定 FAILED（报错 `compiler run returned error code`）。

所以最终方案是让包内**只有一个文件**。

## 构建步骤

```bash
# 0) 前置：本机 venv（torch 2.4.1+cpu 等）在 NN_paper/.venv
VENV=/home/lantuo/Documents/NN_paper/.venv/bin/python
REPO=$(git rev-parse --show-toplevel)   # projective-poly-mnist

# 1) 生成纯矢量 PGF 图（无光栅、无外部文件）
env -u LD_LIBRARY_PATH $VENV paper/tools/make_pgf_figures.py
#    -> /tmp/pgf_out/fig_*.pgf  (10 张，纯矢量)

# 2) 把图内联进单文件版 main.tex
#    输入：已去 bibtex、参考文献内联的 main.tex（见下方"源文件"）
#    输出：/tmp/arxiv_inline/main.tex
python3 paper/tools/inline_figures.py
#    -> /tmp/arxiv_inline/main.tex  (约 613 KB，零外部引用)

# 3) 打包
cd /tmp/arxiv_inline && zip -q -X /tmp/arxiv_submission.zip main.tex
```

## 两个必须注意的技术点

1. **`\pgfsys@defobject` 需要 `@` 是字母。** pgf 代码里的 `\pgfsys@...` 控制词如果在 `\resizebox` 的参数里被 tokenize（此时 `@` 不是字母），会变成 `Undefined control sequence`。做法是用
   ```latex
   \makeatletter
   \newcommand{\inlfigA}{%  <pgf 正文>  }
   \makeatother
   \resizebox{0.86\columnwidth}{!}{\inlfigA}
   ```
   在 `@` 为字母的状态下完成分词。

2. **matplotlib 的 `imshow` 和 `colorbar` 在 pgf 后端会光栅化成 PNG。** 热力图必须改用 `Rectangle` 补丁 + 手绘矢量色条，否则又会出现外部文件依赖。

## 源文件

`inline_figures.py` 读取的是 `/tmp/arxiv_robust/main.tex`（顶层平铺 + 参考文献内联版本，由 `build_final.sh` 流程产出）。如果重建整条链，需先：

1. `paper/main.tex` 的内联版（sections/tables/macros 全展开）
2. 40 条参考文献以 `thebibliography` 内联
3. 所有 `\includegraphics`/`\input` 清空

## 验证

```bash
# 空目录 + 仅 main.tex
mkdir -p /tmp/t && cp /tmp/arxiv_inline/main.tex /tmp/t/ && cd /tmp/t
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
grep -cE '^!' main.log          # 应为 0
pdfinfo main.pdf | grep Pages    # 25 页
```

预期：**exit 0、0 致命错误、0 缺文件、0 悬空引用、0 占位框、25 页**。
