# Polynomial Activations and Projective Normalization

An empirical study of what makes deep fully-connected networks trainable.

**Paper title:** *Polynomial Activations and Projective Normalization: An Empirical Study
of What Makes Deep Fully-Connected Networks Trainable*
**Author:** Xiuyi Yang — fwinlee@gmail.com

This repository contains the complete, self-contained experimental artifact for the paper:
a unified training harness, all 301 experiment configurations, every raw result file, and the
analysis code that turns raw JSON into tables, figures and every number that appears in the
paper. Every number is generated from the raw results — none is typed by hand.

The paper's own LaTeX sources and build products are deliberately **not** part of this
repository; what is published here is the artifact needed to regenerate them.

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

Depth 10, width 128, MNIST. Every number below is reproduced from `experiments/*/*.json`.

| Configuration | Test accuracy (%) |
|---|---|
| Identity, no norm (floor) | 92.71 |
| GELU (best activation) | 98.20 |
| SiLU | 98.13 |
| ReLU (best mainstream, closest rival) | 98.07 |
| tanh | 97.79 |
| Poly `z + λz³`, no norm | 94.34 |
| **Poly + projective `ℓ₂` normalization** | **95.93** |
| Poly + BatchNorm | 96.75 |
| Poly + LayerNorm | 96.29 |
| Poly + RMSNorm | 96.29 |

- **The polynomial alone is not competitive.** The bare cubic trails GELU by **3.86 points**
  at equal parameter count (`\polyNoneToBest`), and the projective normalization recovers
  **1.59** of them (`\polyGap`).
- **Normalization is what unlocks the polynomial.** Projective normalization adds
  **+1.59 points** to the polynomial activation but **−0.19 points** to ReLU: opposite in
  sign and an order of magnitude apart, i.e. a *polynomial-specific* effect.
- **The projection pins the activation scale.** It holds the per-layer activation norm at
  **1×** its initial value where ReLU's decays to **0.048×** and a gated unit's collapses to
  **0** — the mechanism behind the +1.59.
- **Depth is where it matters.** At depth 30 the projected polynomial lands **0.29 points**
  below ReLU, at depth 10 it trails by **2.14 points**; without the projection the polynomial
  collapses to chance (10.28%) at depth 30. At depth 50 *every* design collapses — Poly and
  Poly+`ℓ₂` at 10.28% and 11.35%, and even BatchNorm retains only 44.64%.
- **Gated units need normalization.** SwiGLU/GeGLU/PolyGLU collapse to chance (**11.35%**,
  a **86.6-point** gap) when run bare; with normalization they recover to **97.98%**.
- **Learnable bases buy nothing.** A B-spline activation with BatchNorm reaches **98.36%**,
  but the learned bases show **no consistent gain** over the fixed cubic at equal depth and
  width, while costing appreciably more per step.
- **Faithful reproduction.** The reference configuration reproduces the original standalone
  script bit-for-bit: **96.86%** test accuracy, matching `results_10layers/results.json`.

---

## 2. Repository layout

```
.
├── train_projective_poly_mnist_10layers.py   # original standalone script (reference point)
├── results_10layers/results.json             # its published result: 96.86%
├── run.sh / run_experiments.sh               # wrapper for a single training run
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
└── (paper/ — built locally, not tracked; see "Reproducing" below)
```

Note the three scripts that reference `paper/` write into it as a build-tree, not a
source tree: `analyze.py` and `make_macros.py` create `paper/tables`, `paper/figures`
and `paper/macros.tex` from the raw JSON, so those directories are generated rather than
shipped. `paper/` is listed in `.gitignore`.

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

The repository does **not** ship the MNIST files (they are 12 MB of packed binaries and are
not the interesting artifact). The harness runs fully offline by default (`download=False`)
so that reproduction is network-independent. On a fresh clone, either drop the standard
torchvision `MNIST/raw/` files into `./data/`, or let torchvision fetch them once:

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

# 3. Regenerate every table, figure, and number macro from the raw JSON.
#    These write into paper/ (created if absent, git-ignored).
python experiments/analyze.py                  # -> paper/tables, paper/figures
python experiments/make_macros.py --check      # -> paper/macros.tex + consistency check
```

Step 3 needs no retraining: the 301 raw JSON records are committed, so all tables,
figures and number macros can be regenerated from a clean clone.

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
