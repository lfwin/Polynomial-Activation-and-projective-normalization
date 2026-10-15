"""Does training longer help? Compare 2k vs 20k epochs and lr variants."""
import numpy as np
import torch
from torch import nn

torch.set_default_dtype(torch.float64)

N, d, DEPTH = 64, 5, 10
LAMBDA = 0.1
EPS = 1e-12


def cubic(x):
    return x + LAMBDA * x**3


def projectivize(x):
    return x / torch.linalg.vector_norm(x, dim=-1, keepdim=True).clamp_min(EPS)


class Net(nn.Module):
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
            h = projectivize(cubic(layer(h)))
        return h


def loss_fn(m, X, Y):
    return torch.sum((m(X) - Y) ** 2)


def run(epochs, lr, seed):
    torch.manual_seed(seed)
    X = torch.randn(N, d)
    teacher = Net()
    teacher.eval()
    with torch.no_grad():
        Y = teacher(X).clone()
    student = Net()
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    hist = []
    for ep in range(epochs):
        opt.zero_grad()
        loss = loss_fn(student, X, Y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), max_norm=10.0)
        opt.step()
        hist.append(loss.item())
    return hist


print("seed=7 (the script's seed), vary epochs at lr=1e-3")
for ep in [2000, 5000, 10000, 20000]:
    h = run(ep, 1e-3, 7)
    print(f"  epochs={ep:6d}  final loss={h[-1]:.6e}")

print("\nseed=7, 20000 epochs, vary lr")
for lr in [3e-4, 1e-3, 3e-3]:
    h = run(20000, lr, 7)
    print(f"  lr={lr:.0e}  final loss={h[-1]:.6e}")

print("\n20000 epochs, lr=1e-3, across seeds")
for s in [0, 1, 2, 7, 42]:
    h = run(20000, 1e-3, s)
    print(f"  seed={s:3d}  final loss={h[-1]:.6e}")
