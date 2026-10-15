
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

torch.manual_seed(42)
torch.set_default_dtype(torch.float64)

# Configuration
N = 64
D = 5
DEPTH = 10
LAMBDA = 0.1
EPS = 1e-10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def homogeneous_cubic(s, x, A):
    """
    Projective homogeneous lift of f(t) = t + lambda*t^3.

    Input:  [s, x_1, ..., x_D]
    Output: [s^3, s^2*(A*x) + lambda*(A*x)^3]

    All output coordinates are homogeneous cubics.
    """
    z = x @ A.T
    s_next = s ** 3
    x_next = s.square() * z + LAMBDA * z.pow(3)
    return s_next, x_next


def projectivize(t):
    """
    Rescale each row to the unit sphere.

    Iterating the homogeneous cubic lift without this step sends almost
    every orbit to the hyperplane at infinity (the readout x/s overflows
    within a few layers), so we keep a bounded representative of the
    projective point at every layer.
    """
    norm = torch.linalg.vector_norm(
        t, dim=-1, keepdim=True
    )
    return t / norm.clamp_min(EPS)


class ProjectiveCubicNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.weights = nn.ParameterList([
            nn.Parameter(
                torch.eye(D)
                + 0.03 * torch.randn(D, D)
            )
            for _ in range(DEPTH)
        ])

    def forward(self, X):
        # Initial projective representative [1 : X]
        s = torch.ones(
            X.shape[0], 1, dtype=X.dtype, device=X.device
        )
        x = X

        for A in self.weights:
            s, x = homogeneous_cubic(s, x, A)

            # Reduce to the affine chart, projectivize the point, then
            # re-lift as [1 : t]. This keeps the representative bounded
            # and does not change the network's affine prediction.
            t = projectivize(x / s)
            s = torch.ones_like(s)
            x = t

        # Recover the affine output on the chart s != 0.
        if torch.any(s.abs() < EPS):
            raise FloatingPointError(
                "Output approached the projective hyperplane at infinity."
            )

        return x / s


def make_data():
    X = torch.randn(N, D, device=DEVICE)

    # Teacher-generated targets give a realizable zero-loss problem.
    teacher = ProjectiveCubicNet().to(DEVICE)
    teacher.eval()

    with torch.no_grad():
        Y = teacher(X).clone()

    return X, Y, teacher


def loss_fn(model, X, Y):
    prediction = model(X)
    return torch.sum((prediction - Y) ** 2)


def train(model, X, Y, epochs=2000, lr=1e-4):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    loss_history = []

    for epoch in range(epochs):
        optimizer.zero_grad()

        try:
            loss = loss_fn(model, X, Y)
        except FloatingPointError:
            print("Numerical instability near projective infinity.")
            break

        if not torch.isfinite(loss):
            print("Non-finite loss encountered.")
            break

        loss_history.append(loss.item())

        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=10.0
        )
        optimizer.step()

        if epoch % 100 == 0:
            print(f"epoch={epoch:4d}, loss={loss.item():.8e}")

    return model, loss_history


if __name__ == "__main__":
    X, Y, teacher = make_data()

    student = ProjectiveCubicNet().to(DEVICE)
    student, loss_history = train(student, X, Y)

    student.eval()
    with torch.no_grad():
        final_loss = loss_fn(student, X, Y).item()
        print("Final loss:", final_loss)
        Y_hat = student(X)

    Y_np = Y.cpu().numpy()
    Y_hat_np = Y_hat.cpu().numpy()

    # ========================================================
    # Figure 1: optimization loss curve
    # ========================================================

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(
        np.arange(1, len(loss_history) + 1),
        np.maximum(loss_history, 1e-30),
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Sum-of-squares loss")
    ax.set_title(
        f"Projective cubic network, N={N}, d={D}, depth={DEPTH}"
    )
    ax.grid(True, which="both")
    fig.tight_layout()
    fig.savefig("loss_curve_projectivecubic.png", dpi=160)

    # ========================================================
    # Figure 2: prediction accuracy (target vs prediction)
    # ========================================================

    fig, axes = plt.subplots(1, D, figsize=(16, 3.5))

    for j, ax in enumerate(axes):
        ax.scatter(Y_np[:, j], Y_hat_np[:, j], s=12)
        lo = min(Y_np[:, j].min(), Y_hat_np[:, j].min())
        hi = max(Y_np[:, j].max(), Y_hat_np[:, j].max())
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
        ax.set_title(f"Output {j + 1}")
        ax.set_xlabel("Target")
        ax.set_ylabel("Prediction")
        ax.grid(True)

    fig.tight_layout()
    fig.savefig("prediction_comparison_projectivecubic.png", dpi=160)

    # ========================================================
    # Figure 3: per-sample prediction error
    # ========================================================

    abs_err = np.abs(Y_hat_np - Y_np)

    fig, axes = plt.subplots(1, D, figsize=(16, 3.5))

    for j, ax in enumerate(axes):
        ax.scatter(Y_np[:, j], abs_err[:, j], s=12)
        ax.set_title(f"Output {j + 1}")
        ax.set_xlabel("Target")
        ax.set_ylabel("|error|")
        ax.grid(True)

    fig.tight_layout()
    fig.savefig("prediction_error_projectivecubic.png", dpi=160)

    print(f"Max |error| over all outputs: {abs_err.max():.6e}")

    plt.show()
