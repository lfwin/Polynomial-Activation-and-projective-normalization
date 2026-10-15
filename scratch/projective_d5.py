
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
MAX_NFEV = 300

rng = np.random.default_rng(SEED)


# ============================================================
# 2. Cubic activation and projectivization
# ============================================================

def f(z):
    return z**3 + z


def project(v, eps=1e-12):
    """
    Project each sample vector onto the unit sphere.
    v shape: (N, D)
    """
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.maximum(norms, eps)


# ============================================================
# 3. Three-layer projectivized network
# ============================================================

def forward_projective(X, W, debug=False):
    A1, A2, A3 = W

    # Layer 3
    Z3 = X @ A3.T
    V3 = f(Z3)
    H3 = project(V3)

    # Layer 2
    Z2 = H3 @ A2.T
    V2 = f(Z2)
    H2 = project(V2)

    # Layer 1 / output
    Z1 = H2 @ A1.T
    V1 = f(Z1)
    Yhat = project(V1)

    if debug:
        for name, z in [
            ("Z3", Z3), ("V3", V3), ("H3", H3),
            ("Z2", Z2), ("V2", V2), ("H2", H2),
            ("Z1", Z1), ("V1", V1), ("Yhat", Yhat),
        ]:
            print(
                f"{name:5s}: "
                f"max_abs={np.max(np.abs(z)):.4e}, "
                f"finite={np.isfinite(z).all()}"
            )

    return Yhat


def loss_mse(X, Y, W):
    Yhat = forward_projective(X, W)
    return np.mean((Yhat - Y)**2)


# ============================================================
# 4. Matrix parameterization
# ============================================================

def pack(W):
    return np.concatenate([A.ravel() for A in W])


def unpack(w):
    return [
        w[0:25].reshape(D, D),
        w[25:50].reshape(D, D),
        w[50:75].reshape(D, D),
    ]


def random_matrix():
    # Use the larger matrix scale from your experiment.
    return (
        5.45 * np.eye(D)
        + 0.06 * rng.standard_normal((D, D))
    )


# ============================================================
# 5. Generate zero-loss data
# ============================================================

X = rng.uniform(-10.5, 10.5, size=(N, D))

W_teacher = [
    random_matrix(),
    random_matrix(),
    random_matrix(),
]

Y = forward_projective(X, W_teacher)

print("=" * 70)
print("N =", N, "D =", D)
print("Number of parameters =", 3 * D * D)
print("Teacher projective MSE =", loss_mse(X, Y, W_teacher))

print("\nTeacher network activation diagnostics:")
forward_projective(X, W_teacher, debug=True)

data = np.concatenate([X, Y], axis=1)
columns = (
    [f"x{i+1}" for i in range(D)]
    + [f"y{i+1}" for i in range(D)]
)

pd.DataFrame(data, columns=columns).to_csv(
    "projective_training_data_d5.csv", index=False
)


# ============================================================
# 6. Optimize the projective network
# ============================================================

# Start near the teacher to first test local numerical behavior.
W_initial = [
    A + 1.03 * rng.standard_normal((D, D))
    for A in W_teacher
]


def residual(w):
    W = unpack(w)
    Yhat = forward_projective(X, W)
    return (Yhat - Y).ravel()


print("\nInitial projective MSE:",
      loss_mse(X, Y, W_initial))

result = least_squares(
    residual,
    pack(W_initial),
    method="trf",
    jac="2-point",
    max_nfev=MAX_NFEV,
    ftol=1e-11,
    xtol=1e-11,
    gtol=1e-11,
    verbose=1,
)

W_final = unpack(result.x)
Y_final = forward_projective(X, W_final)

final_loss = np.mean((Y_final - Y)**2)

print("\nOptimization success:", result.success)
print("Message:", result.message)
print("Function evaluations:", result.nfev)
print("Final projective MSE:", final_loss)
print("Residual norm:", np.linalg.norm(result.fun))
print("Gradient infinity norm:",
      np.linalg.norm(result.grad, ord=np.inf))

for i, A in enumerate(W_final):
    print(f"\nA{i+1} final:\n{A}")


# ============================================================
# 7. Save results
# ============================================================

np.savez(
    "projective_parameters_d5.npz",
    A1_teacher=W_teacher[0],
    A2_teacher=W_teacher[1],
    A3_teacher=W_teacher[2],
    A1_final=W_final[0],
    A2_final=W_final[1],
    A3_final=W_final[2],
)

print("\nMaximum prediction error:",
      np.max(np.abs(Y_final - Y)))


# ============================================================
# 8. Plot prediction comparison
# ============================================================

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
plt.savefig("projective_prediction_d5.png", dpi=160)
plt.show()
