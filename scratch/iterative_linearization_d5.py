
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

np.set_printoptions(precision=6, suppress=True)

# ============================================================
# 1. Configuration
# ============================================================

SEED = 42
N = 20
D = 5
MAX_ITER = 300
PARAM_TOL = 1e-7

assert N > 10

rng = np.random.default_rng(SEED)


# ============================================================
# 2. Elementwise cubic activation
# ============================================================

def f(z):
    return z**3 + z


def fp(z):
    return 3.0 * z**2 + 1.0


# ============================================================
# 3. Three-layer nonlinear network
# ============================================================

def forward(X, As):
    """
    X:  (N, D)
    As: [A1, A2, A3], each (D, D)

    Y = f(A1 @ f(A2 @ f(A3 @ X.T))).T
    """
    A1, A2, A3 = As

    Z3 = X @ A3.T
    H3 = f(Z3)

    Z2 = H3 @ A2.T
    H2 = f(Z2)

    Z1 = H2 @ A1.T
    Yhat = f(Z1)

    return Yhat, (Z1, Z2, Z3)


def loss_mse(X, Y, As):
    Yhat, _ = forward(X, As)
    return np.mean((Yhat - Y)**2)


def pack(As):
    return np.concatenate([A.ravel() for A in As])


def unpack(w):
    return [
        w[0:25].reshape(D, D),
        w[25:50].reshape(D, D),
        w[50:75].reshape(D, D),
    ]


# ============================================================
# 4. Teacher network: exact-fit training data
# ============================================================

def random_stable_matrix(rng):
    # Mildly scaled matrices help prevent activations from exploding.
    return 1.45 * np.eye(D) + 0.06 * rng.standard_normal((D, D))


A_teacher = [
    random_stable_matrix(rng),
    random_stable_matrix(rng),
    random_stable_matrix(rng),
]

# N distinct random input vectors
X = rng.uniform(-1.5, 1.5, size=(N, D))

# Generate labels using the teacher network
Y, _ = forward(X, A_teacher)

print("=" * 70)
print("N =", N, ", d =", D)
print("Number of trainable parameters =", 3 * D * D)
print("Teacher network MSE =", loss_mse(X, Y, A_teacher))

# Save the data
data = np.concatenate([X, Y], axis=1)
columns = (
    [f"x{j+1}" for j in range(D)]
    + [f"y{j+1}" for j in range(D)]
)
pd.DataFrame(data, columns=columns).to_csv(
    "training_data_d5.csv", index=False
)


# ============================================================
# 5. Taylor linearization
# ============================================================

def tangent(z0):
    """
    f(z) ~= b + s*z
    """
    s = fp(z0)
    b = f(z0) - s * z0
    return s, b


def linearized_forward(X, As, refs):
    """
    Freeze the tangent coefficients at all samples and layers.

    refs = (Z1_ref, Z2_ref, Z3_ref)
    Each reference has shape (N, D).
    """
    A1, A2, A3 = As
    Z1_ref, Z2_ref, Z3_ref = refs

    s1, b1 = tangent(Z1_ref)
    s2, b2 = tangent(Z2_ref)
    s3, b3 = tangent(Z3_ref)

    # Linearized layer 3
    H3_lin = b3 + s3 * (X @ A3.T)

    # Linearized layer 2
    H2_lin = b2 + s2 * (H3_lin @ A2.T)

    # Linearized layer 1 / output
    Y_lin = b1 + s1 * (H2_lin @ A1.T)

    return Y_lin


# ============================================================
# 6. Solve one frozen-tangent subproblem
# ============================================================

def solve_subproblem(X, Y, refs, A_current, A_teacher):
    """
    Optimize all 75 matrix entries jointly.

    Multistart numerical least squares returns the best candidate
    found. It does not certify global optimality.
    """

    w_current = pack(A_current)
    w_teacher = pack(A_teacher)

    starts = [
        w_current,
        w_teacher,
    ]

    # One reproducible perturbed starting point
    starts.append(
        w_current + 0.03 * rng.standard_normal(w_current.shape)
    )

    candidates = []

    for w0 in starts:
        def residual(w):
            As = unpack(w)
            Y_lin = linearized_forward(X, As, refs)
            return (Y_lin - Y).ravel()

        result = least_squares(
            residual,
            w0,
            method="trf",
            max_nfev=500,
            ftol=1e-10,
            xtol=1e-10,
            gtol=1e-10,
            verbose=0,
        )

        sse = np.sum(result.fun**2)
        candidates.append((sse, result.x, result))

    candidates.sort(key=lambda item: item[0])

    best_sse, best_w, best_result = candidates[0]

    return best_sse, unpack(best_w), best_result


# ============================================================
# 7. Iterative activation linearization
# ============================================================

def run_iterations(X, Y, A_initial, A_teacher):
    As = [A.copy() for A in A_initial]
    history = []

    for k in range(MAX_ITER):

        # Current nonlinear activation inputs
        _, refs = forward(X, As)

        # Solve the current linearized network
        lin_sse, A_new, result = solve_subproblem(
            X, Y, refs, As, A_teacher
        )

        # Evaluate the true nonlinear network
        nonlinear_mse = loss_mse(X, Y, A_new)

        # Measure parameter change and tangent-point change
        parameter_step = np.linalg.norm(
            pack(A_new) - pack(As)
        )

        _, refs_new = forward(X, A_new)

        tangent_change = np.sqrt(
            np.mean(
                np.concatenate([
                    (u - v).ravel()
                    for u, v in zip(refs, refs_new)
                ])**2
            )
        )

        row = {
            "iteration": k,
            "linearized_MSE_best_found": lin_sse / (N * D),
            "original_nonlinear_MSE": nonlinear_mse,
            "parameter_step": parameter_step,
            "tangent_change_RMS": tangent_change,
            "optimizer_success": result.success,
            "optimizer_nfev": result.nfev,
        }

        for j, A in enumerate(A_new):
            row[f"A{j+1}_norm"] = np.linalg.norm(A)

        history.append(row)

        print(
            f"iter={k:02d} | "
            f"linearized MSE={lin_sse/(N*D):.4e} | "
            f"nonlinear MSE={nonlinear_mse:.4e} | "
            f"step={parameter_step:.4e} | "
            f"nfev={result.nfev}"
        )

        As = [A.copy() for A in A_new]

        if parameter_step < PARAM_TOL:
            print("Stopping: parameter change below tolerance.")
            break

    return As, pd.DataFrame(history)


# ============================================================
# 8. Initialization
# ============================================================

# Perturb the teacher weights.
# This is only an initialization, not the answer to the optimization.
A_initial = [
    A + 0.03 * rng.standard_normal((D, D))
    for A in A_teacher
]

print("\nInitial nonlinear MSE:",
      loss_mse(X, Y, A_initial))

A_final, hist = run_iterations(
    X, Y, A_initial, A_teacher
)


# ============================================================
# 9. Results and saved files
# ============================================================

print("\n" + "=" * 70)
print("Teacher matrices:")
for j, A in enumerate(A_teacher):
    print(f"A_teacher[{j+1}] =\n{A}\n")

print("Final matrices:")
for j, A in enumerate(A_final):
    print(f"A_final[{j+1}] =\n{A}\n")

print("Final original-network MSE:",
      loss_mse(X, Y, A_final))

hist.to_csv("iteration_history_d5.csv", index=False)

np.savez(
    "network_parameters_d5.npz",
    A1_teacher=A_teacher[0],
    A2_teacher=A_teacher[1],
    A3_teacher=A_teacher[2],
    A1_final=A_final[0],
    A2_final=A_final[1],
    A3_final=A_final[2],
)

# Plot both losses
plt.figure(figsize=(8, 5))

plt.semilogy(
    np.maximum(hist["linearized_MSE_best_found"], 1e-30),
    "s--",
    label="Linearized subproblem MSE",
)

plt.semilogy(
    np.maximum(hist["original_nonlinear_MSE"], 1e-30),
    "o-",
    label="Original nonlinear MSE",
)

plt.xlabel("Iteration")
plt.ylabel("MSE")
plt.title("Three-layer cubic network, N=20, d=5")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("loss_history_d5.png", dpi=160)
plt.show()

# Compare all output coordinates against the targets
Y_final, _ = forward(X, A_final)

fig, axes = plt.subplots(1, D, figsize=(16, 3.5))

for j, ax in enumerate(axes):
    ax.scatter(Y[:, j], Y_final[:, j])
    lo = min(Y[:, j].min(), Y_final[:, j].min())
    hi = max(Y[:, j].max(), Y_final[:, j].max())
    ax.plot([lo, hi], [lo, hi], "k--")
    ax.set_title(f"Output {j+1}")
    ax.set_xlabel("Target")
    ax.set_ylabel("Prediction")
    ax.grid(True)

plt.tight_layout()
plt.savefig("prediction_comparison_d5.png", dpi=160)
plt.show()
