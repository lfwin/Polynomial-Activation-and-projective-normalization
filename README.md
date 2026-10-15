# Polynomial Activations and Projective Normalization

An empirical study of what makes deep fully-connected networks trainable.

**Paper title:** *Polynomial Activations and Projective Normalization: An Empirical Study
of What Makes Deep Fully-Connected Networks Trainable*
**Author:** Xiuyi Yang — fwinlee@gmail.com

This repository contains the complete, self-contained experimental artifact for the paper:
a unified training harness, all 335 experiment configurations, every raw result file, and the
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

### Depth without normalization: an identity path is enough

The main grid shows the polynomial collapsing to chance at depth 30 *when no normalization
is applied*. The `residual_mnist` group asks the orthogonal question: with the polynomial
but **no normalization of any kind**, does an identity residual path alone buy depth?

| Depth-100, poly, no normalization | Init | Test acc. (%) |
|---|---|---|
| `h^i = f(W^i h^{i-1})` (plain) | xavier | **10.99** (dead) |
| `h^i = f(W^i h^{i-1})` (plain) | identity | 31.38 (partial) |
| `h^i = f(W^i h^{i-1}) + h^{i-1}` (r1) | xavier | **94.50** |
| `h^i = f(W^i h^{i-1}) + h^{i-1}` (r1) | identity | 93.66 |

Extending the depth sweep with `r1` and xavier init: 10 → 97.11, 50 → 93.97, 100 → 94.50,
200 → 94.81, **300 → 93.80** — flat to 300 layers, where the plain arm sits at chance by
depth 50. Two controls in the same group: ReLU + `r1` at depth 100 lands at **94.50**, i.e.
the polynomial with an identity path matches the mainstream activation *without any
normalization*; and renormalizing after the sum (the `r1`-then-normalize variant) falls
straight back to the baseline collapse. The prescription in the paper is in this table:
control scale, and never renormalize after the residual sum.

### Algebraic side: exact critical sets and Bertini decompositions

The small-model claims are computed, not asserted. Exact rational arithmetic (Sage /
SymPy) fixes the dimension and degree of the square-loss critical ideal at one hidden unit
and the real-vs-complex count of its solutions; Bertini then decomposes the critical set
component by component at sampled points, confirming that the loss is constant on each
component and that the number of zero Hessian eigenvalues equals the component's dimension.

```bash
experiments/with_cpu_cap.sh python3 experiments/algebraic_table.sage.py    # dim / degree
experiments/with_cpu_cap.sh python3 experiments/algebraic_real_points.py   # real-vs-complex
experiments/with_cpu_cap.sh python3 experiments/bertini_decomp.py          # components
experiments/with_cpu_cap.sh python3 experiments/loss_surface.py            # loss-surface grid
experiments/with_cpu_cap.sh python3 experiments/init_probe.py             # init assertions
```

`with_cpu_cap.sh` pins these runs to 10 cores. `bertini_decomp.py` shells out to Bertini
(`BERTINI=/path/to/bertini`, default `bertini` on `PATH`) under `taskset`. Single
configurations whose exact computation exceeds 15 minutes are discarded rather than
reported. The solver's scratch directory `experiments/bertini_runs/` is not committed;
`dump_bertini_raw.py` reduces it to the two committed JSON files above.

---

## 2. Repository layout

```
.
├── train_projective_poly_mnist_10layers.py   # original standalone script (reference point)
├── results_10layers/results.json             # its published result: 96.86%
├── run.sh / run_experiments.sh               # wrapper for a single training run
├── experiments/
│   ├── harness.py            # the unified training/eval harness (model, data, metrics)
│   ├── run_all.py            # the full 335-config experiment grid + worker orchestration
│   ├── analyze.py            # raw JSON -> paper/tables/*.tex + paper/figures/*.pdf
│   ├── make_macros.py        # raw JSON -> paper/macros.tex (every number in the paper)
│   ├── verify_fidelity.py    # bit-for-bit check against the original script
│   ├── diag_identity.py      # identity-activation diagnostic
│   ├── with_cpu_cap.sh       # pins the algebraic/numerical probes to 10 cores
│   ├── init_probe.py         # initialization assertions (Jacobian, grad norm, Hessian)
│   ├── loss_surface.py       # square-loss surface grid at tiny width
│   ├── algebraic_table.sage  # exact dim / degree of the critical ideal
│   ├── algebraic_real_points.py  # real vs complex solutions, exact
│   ├── bertini_decomp.py     # Bertini component decomposition + per-component probes
│   ├── dump_bertini_raw.py   # bertini_runs/ scratch -> bertini_*.json
│   ├── status.sh             # progress overview of the result directory
│   ├── logs/                 # raw stdout of the experiment runs (evidence trail)
│   └── <group>/*.json        # 335 raw result records, one file per configuration
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
| `residual_mnist` | 34 | depth 100 identity-residual study + depth sweep to 300, no normalization |
| **total** | **335** | |

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

# 2. Run the full grid (335 configs). Pick --workers to match your core count.
python experiments/run_all.py --group all --workers 8
python experiments/run_all.py --list           # inspect groups without running

# 3. Regenerate every table, figure, and number macro from the raw JSON.
#    These write into paper/ (created if absent, git-ignored).
python experiments/analyze.py                  # -> paper/tables, paper/figures
python experiments/make_macros.py --check      # -> paper/macros.tex + consistency check
```

Step 3 needs no retraining: the 335 raw JSON records are committed, so all tables,
figures and number macros can be regenerated from a clean clone.

The full grid is dominated by single-threaded training runs executed in parallel worker
processes; `--workers 8` is a safe default on a 20-core machine. To re-run one group only
(e.g. after changing one configuration), pass `--group <name>`.

---

## 5. Reproducibility notes

- **All 335 raw records are committed**, so every table and figure can be regenerated
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
