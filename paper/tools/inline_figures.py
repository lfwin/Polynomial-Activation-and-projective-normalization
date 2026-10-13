#!/usr/bin/env python3
"""把 10 张矢量图内联进 main.tex —— 一个文件、零外部依赖、零光栅。
关键：pgf 代码含 \\pgfsys@... 控制词，必须用 \\makeatletter + \\newcommand
在『@ 为字母』的状态下完成分词，再 \\resizebox。"""
import re
from pathlib import Path

SRC  = Path("/tmp/arxiv_robust/main.tex")
PGFD = Path("/tmp/pgf_out")
OUT  = Path("/tmp/arxiv_inline"); OUT.mkdir(exist_ok=True)

main = SRC.read_text()

# 1) preamble
assert '\\usepackage{graphicx}' in main
main = main.replace('\\usepackage{graphicx}',
                    '\\usepackage{graphicx}\n\\usepackage{pgf}', 1)

# 2) 删掉 \safefig 宏与 graphicspath
s = main.index('\\newcommand{\\safefig}')
s = main.rindex('\\makeatletter', 0, s)
e = main.index('\\makeatother', s) + len('\\makeatother')
main = main[:s] + main[e:]
main = main.replace('\\graphicspath{{figures/}{./}}', '')

pat = re.compile(r'\\safefig\[width=([0-9.]+)\\columnwidth\]\{([a-z_]+)\.pdf\}')

def load(name):
    p = PGFD / (name + ".pgf")
    assert p.is_file(), p
    lines = [l for l in p.read_text().splitlines()
             if not l.lstrip().startswith('%%')]
    return "\n".join(lines).strip()

counter = [0]
def repl(m):
    w, name = m.group(1), m.group(2)
    counter[0] += 1
    mac = "\\inlfig" + chr(ord('A') + counter[0] - 1)
    parts = []
    parts.append("\n% ===== figure: " + name + " =====")
    parts.append("\\makeatletter")
    parts.append("\\newcommand{" + mac + "}{%")
    parts.append(load(name))
    parts.append("}")
    parts.append("\\makeatother")
    parts.append("\\resizebox{" + w + "\\columnwidth}{!}{" + mac + "}%")
    return "\n".join(parts) + "\n"

main, n = pat.subn(repl, main)
print("内联图数:", n)
left = re.findall(r'\\input\{|\\includegraphics', main)
print("剩余 input/includegraphics:", left)
assert not left

(OUT / "main.tex").write_text(main)
sz = (OUT / "main.tex").stat().st_size
print("main.tex:", sz, "B =", round(sz / 1024 / 1024, 2), "MB")
