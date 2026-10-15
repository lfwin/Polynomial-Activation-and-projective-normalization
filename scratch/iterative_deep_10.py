
import torch
from torch import nn
import numpy as np
import matplotlib.pyplot as plt

torch.set_default_dtype(torch.float64)
torch.manual_seed(7)

N, d, DEPTH = 64, 5, 10
LAMBDA = 0.1
EPS = 1e-12
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def cubic(x):
    return x + LAMBDA * x**3


def projectivize(x):
    norm = torch.linalg.vector_norm(
        x, dim=-1, keepdim=True
    )
    return x / norm.clamp_min(EPS)


class CubicProjectiveNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Linear(d, d, bias=False)
            for _ in range(DEPTH)
        ])

        # Initialize close to identity
        with torch.no_grad():
            for layer in self.layers:
                layer.weight.copy_(
                    torch.eye(d)
                    + 0.03 * torch.randn(d, d)
                )

    def forward(self, x):
        h = projectivize(x)

        for layer in self.layers:
            h = layer(h)
            h = cubic(h)
            h = projectivize(h)

        return h


# Generate N > 50 samples
X = torch.randn(N, d, device=DEVICE)

# Teacher generates exact zero-loss targets
teacher = CubicProjectiveNet().to(DEVICE)
teacher.eval()

with torch.no_grad():
    Y = teacher(X).clone()

# Independently initialized student
student = CubicProjectiveNet().to(DEVICE)

def loss_fn(model, X, Y):
    residual = model(X) - Y
    return torch.sum(residual**2)


optimizer = torch.optim.Adam(
    student.parameters(), lr=1e-3
)

loss_history = []

for epoch in range(2000):
    optimizer.zero_grad()
    loss = loss_fn(student, X, Y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(
        student.parameters(), max_norm=10.0
    )
    optimizer.step()

    loss_history.append(loss.item())

    if epoch % 100 == 0:
        print(
            f"epoch={epoch:4d}, "
            f"loss={loss.item():.12e}"
        )

print("Final loss:", loss_fn(student, X, Y).item())

# ============================================================
# Final predictions and per-sample error
# ============================================================

student.eval()
with torch.no_grad():
    Y_hat = student(X)

Y_np = Y.cpu().numpy()
Y_hat_np = Y_hat.cpu().numpy()

# ============================================================
# Figure 1: optimization loss curve
# ============================================================

fig, ax = plt.subplots(figsize=(8, 5))
ax.semilogy(
    np.arange(1, len(loss_history) + 1),
    np.maximum(loss_history, 1e-30),
)
ax.set_xlabel("Epoch")
ax.set_ylabel("Sum-of-squares loss")
ax.set_title(
    f"Cubic projective network, N={N}, d={d}, depth={DEPTH}"
)
ax.grid(True, which="both")
fig.tight_layout()
fig.savefig("loss_curve_deep10.png", dpi=160)

# ============================================================
# Figure 2: prediction accuracy (target vs prediction)
# ============================================================

fig, axes = plt.subplots(1, d, figsize=(16, 3.5))

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
fig.savefig("prediction_comparison_deep10.png", dpi=160)

# ============================================================
# Figure 3: per-sample prediction error
# ============================================================

abs_err = np.abs(Y_hat_np - Y_np)

fig, axes = plt.subplots(1, d, figsize=(16, 3.5))

for j, ax in enumerate(axes):
    ax.scatter(Y_np[:, j], abs_err[:, j], s=12)
    ax.set_title(f"Output {j + 1}")
    ax.set_xlabel("Target")
    ax.set_ylabel("|error|")
    ax.grid(True)

fig.tight_layout()
fig.savefig("prediction_error_deep10.png", dpi=160)

max_err = abs_err.max()
print(f"Max |error| over all outputs: {max_err:.6e}")

plt.show()
