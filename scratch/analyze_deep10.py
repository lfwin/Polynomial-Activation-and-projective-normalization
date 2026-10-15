"""Deeper diagnostics for iterative_deep_10.py (does not modify it)."""
import numpy as np
import torch
from torch import nn

torch.set_default_dtype(torch.float64)
torch.manual_seed(7)

N, d, DEPTH = 64, 5, 10
LAMBDA = 0.1
EPS = 1e-12
DEVICE = "cpu"


def cubic(x):
    return x + LAMBDA * x**3


def projectivize(x):
    norm = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
    return x / norm.clamp_min(EPS)


class CubicProjectiveNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList(
            [nn.Linear(d, d, bias=False) for _ in range(DEPTH)]
        )
        with torch.no_grad():
            for layer in self.layers:
                layer.weight.copy_(torch.eye(d) + 0.03 * torch.randn(d, d))

    def forward(self, x):
        h = projectivize(x)
        for layer in self.layers:
            h = layer(h)
            h = cubic(h)
            h = projectivize(h)
        return h


def loss_fn(model, X, Y):
    return torch.sum((model(X) - Y) ** 2)


X = torch.randn(N, d, device=DEVICE)
teacher = CubicProjectiveNet().to(DEVICE)
teacher.eval()
with torch.no_grad():
    Y = teacher(X).clone()

student = CubicProjectiveNet().to(DEVICE)
optimizer = torch.optim.Adam(student.parameters(), lr=1e-3)

hist = []
for epoch in range(2000):
    optimizer.zero_grad()
    loss = loss_fn(student, X, Y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=10.0)
    optimizer.step()
    hist.append(loss.item())

hist = np.array(hist)
student.eval()
with torch.no_grad():
    Y_hat = student(X)
    Y_t = teacher(X)

res = (Y_hat - Y).cpu().numpy()
Yn = Y.cpu().numpy()

print("=== convergence shape ===")
for a, b in [(0, 100), (100, 500), (500, 1000), (1000, 1500), (1500, 2000)]:
    seg = hist[a:b]
    ratio = seg[-1] / seg[0]
    print(f"epoch {a:4d}-{b:4d}: loss {seg[0]:.3e} -> {seg[-1]:.3e}  "
          f"(x{ratio:.3f}, log-slope {np.log(ratio)/ (b-a):+.2e}/ep)")

# Fit log(loss) vs epoch over last 1000 epochs
e = np.arange(1000, 2000)
slope = np.polyfit(e, np.log(hist[1000:2000]), 1)[0]
print(f"late-phase log-loss slope: {slope:+.3e} per epoch "
      f"(=> ~{np.log(10)/ -slope:,.0f} epochs to drop 10x)" if slope < 0
      else f"late-phase log-loss slope: {slope:+.3e} per epoch")

print("\n=== error structure ===")
mse_per_dim = (res**2).mean(axis=0)
print("per-output-dim MSE:", np.array2string(mse_per_dim, precision=3, floatmode="maxprec"))
print(f"mean |residual|={np.abs(res).mean():.3e}  max |residual|={np.abs(res).max():.3e}")
print(f"residual RMS={np.sqrt((res**2).mean()):.3e}")
print(f"target RMS={np.sqrt((Yn**2).mean()):.3e}")
print(f"relative RMS error = {np.sqrt((res**2).mean())/np.sqrt((Yn**2).mean()):.3e}")

# correlation between residual and target per dim (systematic bias?)
print("\n=== residual bias check (corr of residual with target, per dim) ===")
for j in range(d):
    c = np.corrcoef(res[:, j], Yn[:, j])[0, 1]
    print(f"  dim {j+1}: corr={c:+.3f}  mean(res)={res[:, j].mean():+.2e}")

# teacher vs student weight mismatch (norms)
print("\n=== parameter comparison (weights differ due to projective gauge) ===")
for i, (lt, ls) in enumerate(zip(teacher.layers, student.layers)):
    wt = lt.weight.detach().numpy()
    ws = ls.weight.detach().numpy()
    print(f"  layer {i:2d}: ||W_teacher||={np.linalg.norm(wt):.4f}  "
          f"||W_student||={np.linalg.norm(ws):.4f}  "
          f"||dW||={np.linalg.norm(ws-wt):.4f}")

# output norm check: outputs live on unit sphere
print("\n=== output geometry ===")
print(f"||Y|| row norms: min={np.linalg.norm(Yn,axis=1).min():.6f} "
      f"max={np.linalg.norm(Yn,axis=1).max():.6f}")
Yh = Y_hat.cpu().numpy()
print(f"||Yhat|| row norms: min={np.linalg.norm(Yh,axis=1).min():.6f} "
      f"max={np.linalg.norm(Yh,axis=1).max():.6f}")

# how well can we do? nearest-neighbor / random baseline
print("\n=== baselines ===")
print(f"loss(final)      = {hist[-1]:.6e}")
print(f"loss(init)       = {hist[0]:.6e}")
print(f"loss if predict all-zeros = {np.sum(Yn**2):.6e}")
print(f"loss if predict mean(Y)   = {np.sum((Yn - Yn.mean(0))**2):.6e}")
