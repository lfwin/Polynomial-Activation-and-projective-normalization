# scratch/

Early exploration scripts that predate the cleaned-up pipeline in `experiments/`.
Kept for provenance; the paper's numbers all come from `experiments/`, not from
these files.

- `projective_cubic_critical.sage` — exact critical ideal for the N=3, d=2,
  depth-3 projective cubic toy model (SageMath).
- `one_layer_3_polynomial_system.sage`, `rankdrop.sage` — exact one-layer
  polynomial systems and the rank-drop ideal over Q(sqrt(2)).
- `ProjectiveCubicNet.py`, `projective_d5.py` — torch reference models for the
  toy systems.
- `iterative_linearization.py`, `iterative_linearization_d5.py`,
  `iterative_deep_10.py` — iterative-linearization training probes.
- `analyze_deep10.py`, `analyze_deep10_longrun.py`, `compare_two_nets.py` —
  result comparison helpers for the early deep-10 runs.

Binary `.sobj` artifacts and preparsed `.sage.py` files are not committed.
