# Polynomial Activations and Projective Normalization

An empirical study of what makes deep fully-connected networks trainable.

**Paper title:** *Polynomial Activations and Projective Normalization: An Empirical Study
of What Makes Deep Fully-Connected Networks Trainable*
**Author:** Xiuyi Yang — fwinlee@gmail.com

This repository contains the complete, self-contained experimental artifact for the paper:
a unified training harness, all 301 experiment configurations, every raw result file, the
analysis code that turns raw JSON into the paper's tables and figures, and the LaTeX source.
Every number in the paper is generated from the raw results — none is typed by hand.

---

## 1. What the paper asks

Two design choices are compared head-to-head on MNIST (and on a set of analytic synthetic
tasks), holding architecture and optimisation fixed:

1. **Activation function** — classical polynomials vs. mainstream activations
   (ReLU, GELU, SiLU, tanh, SwiGLU, GEGLU).
2. **Normalization layer** — none, L2, BatchNorm, LayerNorm, RMSNorm, and a
   **projective normalization** that divides the pre-activation by its own norm.

The framing claim is that these two are *not* independent knobs: the polynomial activation's
benefit is realised only once a normalization layer is in place.

### Headline results (all reproduced from `experiments/*/**.json`)

| Setting (depth 10, width 128, MNIST) | Test accuracy |
|---|---|
| Identity (no activation), no norm | 92.71 |
| Polynomial, no norm | 94.34 |
| **Polynomial + projective normalization** | **95.93** |
| ReLU (best mainstream) | 98.07 |
| GELU | 98.20 |
| SiLU | 98.13 |

- **Activation band is narrow.** The best mainstream activations sit within **0.13 points**
  of each other — the choice among them matters little.
- **Normalization is what unlocks the polynomial.** Projective normalization adds
  **+1.59 points** to the polynomial activation, but **−0.19 points** to ReLU: opposite in
  sign and an order of magnitude apart, i.e. it is a *polynomial-specific* effect.
- **Gated units are useless without normalization.** SwiGLU/GEGLU/poly-GLU all collapse to
  chance (11.35%) when run bare; with LayerNorm they recover to ~97.98%.
- **Learnable polynomials win.** A B-spline activation with BatchNorm reaches **98.36%**,
  above GELU's 98.20%.
- **Depth.** Projective normalization drives the depth behaviour, not the polynomial: at
  depth 30 polynomial+projective (94.10) is level with ReLU (94.39); at depth 50 both
  collapse (11.35%).
- **Faithful reproduction.** The reference configuration reproduces the original standalone
  script bit-for-bit: **96.86%** test accuracy, matching `results_10layers/results.json`.

---

## 2. Repository layout

```
.
├── train_projective_poly_mnist_10layers.py   # original standalone script (reference point)
├── results_10layers/results.json             # its published result: 96.86%
├── run.sh / run_experiments.sh               # conveniences wrappers
├── make_paper.sh / finish_paper.sh           # one-shot regenerate-tables-and-build-paper
├── experiments/
│   ├── harness.py            # the unified training/eval harness (model, data, metrics)
│   ├── run_all.py            # the full 301-config experiment grid + worker orchestration
│   ├── analyze.py            # raw JSON -> paper/tables/*.tex + paper/figures/*.pdf
│   ├── make_macros.py        # raw JSON -> paper/macros.tex (every number in the paper)
│   ├── verify_fidelity.py    # bit-for-bit check against the original script
│   ├── diag_identity.py      # identity-activation diagnostic
│   ├── status.sh             # progress overview of the result directory
│   ├── logs/                 # raw stdout of the experiment runs (evidence trail)
│   └── <group>/*.json        # 301 raw result records, one file per configuration
└── paper/
    ├── main.tex / refs.bib / main.bbl        # LaTeX source (arXiv-style, single column)
    ├── macros.tex                            # 188 generated number macros
    ├── sections/*.tex                        # 01_intro .. 07_conclusion
    ├── tables/*.tex                          # generated tables
    ├── figures/*.pdf                         # generated figures
    └── main.pdf                              # compiled paper (22 pages)
```

Result groups under `experiments/`:

| Group | Configs | What it covers |
|---|---|---|
| `act_mnist` | 10 | activation sweep on MNIST at fixed depth/width |
| `norm_mnist` | 20 | normalization sweep, including the projective variant |
| `depth_mnist` | 42 | depth 2–20 |
| `depth_very` | 14 | depth 30 and 50 (failure boundary) |
| `width_mnist` | 12 | width sweep |
| `degree_mnist` | 16 | polynomial degree / λ grid |
| `learn_poly_mnist` | 16 | learnable polynomial families (B-spline, Chebyshev, Legendre) |
| `glu_norm_mnist` | 30 | gated units with and without normalization |
| `seeds` | 25 | 5-seed variance study |
| `ref_orig` / `ref_fidelity` | 1 + 1 | faithful reproduction of the original script |
| `synth3` / `synth3_depth` / `synth3_norm` | 64 + 18 + 32 | synthetic diagnostic tasks |
| **total** | **301** | |

---

## 3. Setup

Requires Python 3.8+ and a **CPU** build of PyTorch.

```bash
python3 -m venv .venv
source .venv/bin/activate

# CPU wheels — do NOT use the default CUDA index for these versions
pip install torch==2.4.1 torchvision==0.19.1 \
    --index-url https://download.pytorch.org/whl/cpu
pip install numpy matplotlib

# for building the paper
sudo apt-get install texlive-latex-base texlive-latex-extra texlive-fonts-recommended
```

> **Note on `LD_LIBRARY_PATH`.** If your shell exports an `LD_LIBRARY_PATH` pointing at a C++
> libtorch install, it will shadow the pip-installed torch and `import torch` will segfault.
> Run everything with it cleared:
> `env -u LD_LIBRARY_PATH python ...`

### MNIST data

The repository does **not** ship the MNIST files. The harness runs fully offline by default
(`download=False`) so that reproduction is network-independent — the packed data
itself is the original raw files. On a fresh clone, either drop the standard torchvision
`MNIST/raw/` files into `./data/`, or let torchvision fetch them once:

```bash
python experiments/harness.py --dataset mnist --data-dir ./data --allow-download ...
```

---

## 4. Reproducing

Every script uses only relative paths and the interpreter you invoke it with. Nothing is
hardcoded to a machine.

```bash
# 1. Confirm the harness reproduces the original standalone script bit-for-bit
python experiments/verify_fidelity.py          # expect 96.86%

# 2. Run the full grid (301 configs). Pick --workers to match your core count.
python experiments/run_all.py --group all --workers 8
python experiments/run_all.py --list           # inspect groups without running

# 3. Regenerate every table, figure, and number macro from the raw JSON
python experiments/analyze.py                  # -> paper/tables, paper/figures
python experiments/make_macros.py --check      # -> paper/macros.tex + consistency check

# 4. Build the paper
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
# or, from the repo root, the whole pipeline in one shot:
./make_paper.sh
```

`make_paper.sh` runs steps 3 and 4 together and rebuilds `paper/main.pdf`.

The full grid is dominated by single-threaded training runs executed in parallel worker
processes; `--workers 8` is a safe default on a 20-core machine. To re-run one group only
(e.g. after changing one configuration), pass `--group <name>`.

---

## 5. Reproducibility notes

- **All 301 raw records are committed**, so every table and figure can be regenerated
  without retraining. `analyze.py` and `make_macros.py` read the JSON directly.
- **No hand-written numbers in the paper.** The prose cites macros such as
  `\polyProjAcc` (95.93) and `\polyGap` (1.59) that are generated by `make_macros.py`;
  `make_macros.py --check` fails loudly if any macro cannot be resolved from the raw data.
- **Bit-for-bit fidelity.** `harness.py --legacy-loader --no-diagnostics` is the exact
  condition under which the unified harness reproduces the original script's
  data order and therefore its accuracy to the digit. `verify_fidelity.py` encodes this.
- **Synthetic tasks are analytic**, generated from a fixed seed family, so they need no
  data download at all and are the fastest way to smoke-test the pipeline.

## 6. Citation

```bibtex
@misc{yang2026polynomial,
  title  = {Polynomial Activations and Projective Normalization:
            An Empirical Study of What Makes Deep Fully-Connected Networks Trainable},
  author = {Xiuyi Yang},
  year   = {2026},
  note   = {fwinlee@gmail.com}
}
```

## 7. License

MIT — see `LICENSE`.
