"""Head-to-head comparison of the two network implementations.

  deep10  = iterative_deep_10.py       (CubicProjectiveNet)
  proj    = ProjectiveCubicNet.py      (ProjectiveCubicNet, after the fix)

Goal: isolate exactly what differs (input normalization) and measure the
effect on training.
"""
import numpy as np
import torch
from torch import nn

torch.set_default_dtype(torch.float64)

N, D, DEPTH, LAM, EPS = 64, 5, 10, 0.1, 1e-12


def norm(v):
    return v / torch.linalg.vector_norm(v, dim=-1, keepdim=True).clamp_min(EPS)


def f(z):
    return z + LAM * z**3


# ---- deep10 forward: u0 = P(X), u_{k+1} = P(f(A u_k)) ----
def forward_deep10(W, X):
    h = norm(X)
    for A in W:
        h = norm(f(h @ A.T))
    return h


# ---- proj forward (fixed): homogeneous lift, s reset to 1 each layer ----
def forward_proj(W, X):
    s = torch.ones(X.shape[0], 1, dtype=X.dtype)
    x = X.clone()
    for A in W:
        z = x @ A.T
        s_next = s**3
        x_next = s.square() * z + LAM * z.pow(3)
        t = norm(x_next / s_next)
        s = torch.ones_like(s)
        x = t
    return x / s


def make_net(seed):
    torch.manual_seed(seed)
    return [torch.eye(D) + 0.03 * torch.randn(D, D) for _ in range(DEPTH)]


def train(W_init, X, Y, lr, epochs):
    W = [w.clone().requires_grad_(True) for w in W_init]
    opt = torch.optim.Adam(W, lr=lr)
    hist = []
    for _ in range(epochs):
        opt.zero_grad()
        loss = torch.sum((forward_proj(W, X) - Y) ** 2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(W, 10.0)
        opt.step()
        hist.append(loss.item())
    return W, hist


print("=" * 68)
print("1. Are the two forward maps the same function?")
torch.manual_seed(42)
W = make_net(42)
X = torch.randn(N, D)
a = forward_deep10(W, X)
b = forward_proj(W, X)
c = forward_proj(W, norm(X))   # proj, but with the same input normalization deep10 uses
print(f"   deep10 vs proj(raw X)      : max|diff| = {(a - b).abs().max().item():.3e}")
print(f"   deep10 vs proj(normalized) : max|diff| = {(a - c).abs().max().item():.3e}")
print("   -> identical recurrence u_{k+1}=P(f(A u_k)); the ONLY gap is")
print("      whether the input is projected onto the sphere first.")

print("\n" + "=" * 68)
print("2. Training comparison (same data, same init, same lr/epochs)")


def compare(lr, epochs=2000, data_seed=42, init_seed=123):
    torch.manual_seed(data_seed)
    Xr = torch.randn(N, D)
    Wt = make_net(data_seed + 1000)
    Y_deep = forward_deep10(Wt, Xr)
    Y_proj = forward_proj(Wt, Xr)

    W0 = make_net(init_seed)
    _, hd = train(W0, Xr, Y_deep, lr, epochs)
    _, hp = train(W0, Xr, Y_proj, lr, epochs)
    return hd[-1], hp[-1]


for lr in [1e-4, 1e-3]:
    d10, pr = compare(lr)
    print(f"   lr={lr:g}:  deep10 final={d10:.6e}   proj final={pr:.6e}")

print("\n" + "=" * 68)
print("3. Input-normalization ablation for proj (does it matter?)")


def train_variant(normalize_input, lr=1e-3, epochs=2000):
    torch.manual_seed(42)
    Xr = torch.randn(N, D)
    Wt = make_net(1042)
    if normalize_input:
        Y = forward_deep10(Wt, Xr)
    else:
        Y = forward_proj(Wt, Xr)
    W0 = make_net(123)

    def fwd(W, X):
        return forward_deep10(W, X) if normalize_input else forward_proj(W, X)

    W = [w.clone().requires_grad_(True) for w in W0]
    opt = torch.optim.Adam(W, lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        loss = torch.sum((fwd(W, Xr) - Y) ** 2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(W, 10.0)
        opt.step()
    return loss.item()


print(f"   proj with input-norm    : {train_variant(True):.6e}")
print(f"   proj without input-norm : {train_variant(False):.6e}")
