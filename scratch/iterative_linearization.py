
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

np.set_printoptions(precision=12, suppress=True)

# ============================================================
# 1. Network and zero-loss simulation data
# ============================================================

N = 20                       # N > 10
assert N > 10

def f(z):
    return z**3 + z

def fp(z):
    return 3.0 * z**2 + 1.0

def forward(x, w):
    """
    w = [a1, a2, a3]
    y = f(a1 * f(a2 * f(a3*x)))
    """
    a1, a2, a3 = w

    z3 = a3 * x
    h3 = f(z3)

    z2 = a2 * h3
    h2 = f(z2)

    z1 = a1 * h2
    yhat = f(z1)

    return yhat, (z1, z2, z3)


# Generate exact-fit teacher data
x = np.linspace(-0.2, 0.2, N)
w_teacher = np.array([0.8, 0.9, 1.1])

y, _ = forward(x, w_teacher)

print("=" * 65)
print("Teacher parameters:", w_teacher)
print("Initial data loss:",
      np.mean((forward(x, w_teacher)[0] - y)**2))

pd.DataFrame({"x": x, "y": y}).to_csv(
    "training_data.csv", index=False
)


# ============================================================
# 2. First-order Taylor linearization
# ============================================================

def tangent(z0):
    """
    f(z) ~= b + s*z
    """
    s = fp(z0)
    b = f(z0) - s * z0
    return s, b


def linearized_forward(x, w, refs):
    """
    Freeze the activation input of every layer for every sample.
    Optimize weights while keeping these tangent coefficients fixed.
    """
    a1, a2, a3 = w
    z1_ref, z2_ref, z3_ref = refs

    s1, b1 = tangent(z1_ref)
    s2, b2 = tangent(z2_ref)
    s3, b3 = tangent(z3_ref)

    # Layer 3
    h3_lin = b3 + s3 * (a3 * x)

    # Layer 2
    h2_lin = b2 + s2 * (a2 * h3_lin)

    # Layer 1 / output
    y_lin = b1 + s1 * (a1 * h2_lin)

    return y_lin


# ============================================================
# 3. Solve the fixed-linearization subproblem
# ============================================================

def solve_linearized_subproblem(x, y, refs, w_current):
    """
    Multistart numerical optimization.

    Important:
    This returns the best solution found among the starts.
    It does NOT certify global optimality.
    """
    starts = [
        w_current.copy(),
        np.array([0.5, 0.5, 0.5]),
        np.array([1.0, 1.0, 1.0]),
        np.array([0.8, 0.9, 1.1]),
        np.array([-0.8, 0.9, 1.1]),
    ]

    candidates = []

    for w0 in starts:
        result = least_squares(
            lambda w: linearized_forward(x, w, refs) - y,
            w0,
            max_nfev=10000,
            ftol=1e-13,
            xtol=1e-13,
            gtol=1e-13,
        )

        sse = np.sum(result.fun**2)
        candidates.append((sse, result.x, result))

    candidates.sort(key=lambda item: item[0])
    return candidates[0]


# ============================================================
# 4. Iteratively update all three layers
# ============================================================

def run_iterations(
    x,
    y,
    w0,
    max_iter=30,
    parameter_tol=1e-9,
):
    w = np.asarray(w0, dtype=float).copy()
    history = []

    for k in range(max_iter):

        # Current nonlinear network determines the tangent points
        _, refs = forward(x, w)

        # Solve the frozen-tangent network
        lin_sse, w_new, result = solve_linearized_subproblem(
            x, y, refs, w
        )

        # Evaluate the ORIGINAL nonlinear network
        y_new, new_refs = forward(x, w_new)
        nonlinear_sse = np.sum((y_new - y)**2)

        # Evaluate tangent points after the update
        tangent_change = np.sqrt(sum(
            np.mean((u - v)**2)
            for u, v in zip(refs, new_refs)
        ))

        history.append({
            "iteration": k,
            "a1": w_new[0],
            "a2": w_new[1],
            "a3": w_new[2],
            "linearized_MSE_best_found": lin_sse / len(x),
            "nonlinear_MSE": nonlinear_sse / len(x),
            "parameter_step": np.linalg.norm(w_new - w),
            "tangent_change_RMS": tangent_change,
            "optimizer_success": result.success,
            "optimizer_message": result.message,
        })

        print(
            f"iter={k:02d} | "
            f"w={w_new} | "
            f"linearized MSE={lin_sse/len(x):.4e} | "
            f"original MSE={nonlinear_sse/len(x):.4e}"
        )

        step = np.linalg.norm(w_new - w)
        w = w_new.copy()

        if step < parameter_tol:
            print("Stopping: parameter change is small.")
            break

    return w, pd.DataFrame(history)


# ============================================================
# 5. Run the experiment
# ============================================================

# Start near, but not exactly at, the teacher parameters.
w_initial = np.array([0.75, 0.85, 1.05])

print("\nInitial parameters:", w_initial)
print("Initial nonlinear MSE:",
      np.mean((forward(x, w_initial)[0] - y)**2))

w_final, hist = run_iterations(
    x, y, w_initial, max_iter=30
)

print("\n" + "=" * 65)
print("Teacher parameters:", w_teacher)
print("Initial parameters:", w_initial)
print("Final parameters:", w_final)

y_final, _ = forward(x, w_final)
print("Final original-network MSE:",
      np.mean((y_final - y)**2))

hist.to_csv("iteration_history.csv", index=False)

# ============================================================
# 6. Plot original loss and linearized subproblem loss
# ============================================================

plt.figure()
plt.semilogy(
    np.maximum(hist["nonlinear_MSE"].to_numpy(), 1e-30),
    "o-", label="Original nonlinear MSE"
)
plt.semilogy(
    np.maximum(
        hist["linearized_MSE_best_found"].to_numpy(), 1e-30
    ),
    "s--", label="Linearized subproblem MSE"
)
plt.xlabel("Iteration")
plt.ylabel("MSE (log scale)")
plt.title("Iterative activation linearization")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("loss_history.png", dpi=160)
plt.show()

# Compare predicted and target values
plt.figure()
plt.plot(x, y, "o", label="Target")
plt.plot(x, y_final, "-", label="Final nonlinear prediction")
plt.xlabel("x")
plt.ylabel("y")
plt.title("Teacher data vs final nonlinear network")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.savefig("prediction_comparison.png", dpi=160)
plt.show()
