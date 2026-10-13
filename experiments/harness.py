#!/usr/bin/env python3
"""
统一实验框架：激活函数 x 归一化方式 x 深度 x 数据集。

设计目标
--------
1) 严格复现 train_projective_poly_mnist_10layers.py（--activation poly_proj
   --norm none，或等价的 --activation poly --norm l2 --norm-pos post）。
2) 把"多项式激活"与"射影 L2 归一化"解耦，从而能做消融。
3) 覆盖深度学习/大模型常用激活函数（ReLU/LeakyReLU/ELU/SELU/GELU/SiLU/Mish/
   tanh/sigmoid/Softplus/Identity）与门控变体（SwiGLU/GeGLU/ReGLU/
   Bilinear/SELU-GLU/Mish-GLU/Tanh-GLU），以及归一化层（BatchNorm/LayerNorm/
   RMSNorm/射影 L2）。
4) 支持自构数据集（rings/spiral/xor/checker/blobs/parity/cubes）。

每个配置输出一个 JSON 到 --out-dir，可并行运行、可断点续跑。
"""
import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms


# ============================================================
# 激活函数
# ============================================================

class ProjectivePolynomial(nn.Module):
    """f(z) = z + lam * z^degree，截断在 [-clamp, clamp]。

    normalize=True 时内置 L2 射影归一化，等价于原代码的
    ProjectivePolynomialActivation；normalize=False 时只是纯多项式。
    """

    def __init__(self, degree=3, lam=0.01, clamp=10.0, eps=1e-8,
                 normalize=False):
        super().__init__()
        self.degree = degree
        self.lam = lam
        self.clamp = clamp
        self.eps = eps
        self.normalize = normalize

    def forward(self, z):
        z = z.clamp(-self.clamp, self.clamp)
        y = z + self.lam * z.pow(self.degree)
        if self.normalize:
            norm = torch.linalg.vector_norm(y, ord=2, dim=1, keepdim=True)
            y = y / norm.clamp_min(self.eps)
        return y

    def extra_repr(self):
        return (f"degree={self.degree}, lam={self.lam}, "
                f"normalize={self.normalize}")


def make_activation(name, degree=3, lam=0.01, lpoly_degree=6, bspline_grid=8):
    """逐元素激活。'poly' 不含 L2 归一化（归一化由 norm 参数控制）。"""
    name = name.lower()
    table = {
        "identity": lambda: nn.Identity(),
        "relu": lambda: nn.ReLU(),
        "gelu": lambda: nn.GELU(approximate="none"),
        "silu": lambda: nn.SiLU(),
        "tanh": lambda: nn.Tanh(),
        "poly": lambda: ProjectivePolynomial(degree=degree, lam=lam,
                                             normalize=False),
        "poly_proj": lambda: ProjectivePolynomial(degree=degree, lam=lam,
                                                  normalize=True),
        # 可学习多项式族（"不同多项式激活函数"的学习版对照）
        "legendre": lambda: LegendrePoly(degree=lpoly_degree, B=4.0),
        "bspline": lambda: BSpline(grid=bspline_grid, degree=3, B=2.0,
                                   base_act="silu"),
        "chebyshev": lambda: LegendrePoly(degree=lpoly_degree, B=4.0,
                                          basis="chebyshev"),
    }
    if name not in table:
        raise ValueError(f"未知激活函数: {name}")
    return table[name]()


class Sin(nn.Module):
    """SIREN 式正弦激活：sin(omega * z)，omega 固定为 1（可由输入尺度吸收）。"""

    def __init__(self, omega=1.0):
        super().__init__()
        self.omega = omega

    def forward(self, z):
        return torch.sin(self.omega * z)


class LegendrePoly(nn.Module):
    """可学习多项式激活（Legendre 基），是 z + lambda z^3 的学习式推广。

        f(z) = gain * sum_{k=0}^{K} c_k * L_k(clamp(z)/B)

    初始 c_1 = 1、其余为 0、gain = 1，即从恒等映射出发；训练后可拟合任意
    低阶多项式（含立方项），因此与固定立方激活构成"固定 vs 可学习"的对照。
    """

    SCALE = 16.0   # 归一化区间的视觉尺度（原文 clamp 到 10，这里取 16 便于
    B = 4.0        # 实际激活幅度落在可分辨的区间内）

    def __init__(self, degree=6, B=4.0, gain=1.0, scale=16.0,
                 basis="legendre"):
        super().__init__()
        self.degree = degree
        self.B = B
        self.scale = scale
        self.basis_name = basis
        self.coeff = nn.Parameter(torch.zeros(degree + 1))
        with torch.no_grad():
            self.coeff[1] = 1.0
        self.gain = nn.Parameter(torch.tensor(float(gain)))

    def _legendre(self, u):
        """u ∈ [-1,1]；返回 [L_0(u), ..., L_K(u)] 列表。"""
        P = [torch.ones_like(u), u.clone()]
        for k in range(2, self.degree + 1):
            P.append(((2 * k - 1) * u * P[-1] - (k - 1) * P[-2]) / k)
        return P[:self.degree + 1]

    def _chebyshev(self, u):
        """第一类 Chebyshev 多项式 T_k(u)，u ∈ [-1,1]。"""
        P = [torch.ones_like(u), u.clone()]
        for k in range(2, self.degree + 1):
            P.append(2 * u * P[-1] - P[-2])
        return P[:self.degree + 1]

    def forward(self, z):
        u = (z / self.scale).clamp(-self.B, self.B) / self.B
        P = (self._chebyshev(u) if self.basis_name == "chebyshev"
             else self._legendre(u))
        out = 0.0
        for k, Pk in enumerate(P):
            out = out + self.coeff[k] * Pk
        return self.gain * out

    def extra_repr(self):
        return (f"degree={self.degree}, B={self.B}, basis={self.basis_name}")


class BSpline(nn.Module):
    """KAN 风格的三次 B-spline 激活（可学习控制点）。

        phi(z) = w_base * SiLU(z) + w_spline * sum_j c_j * N_j(z)

    N_j 为均匀节点上的三次 B-spline 基（G 个区间、p=3）。控制点初始化为 0、
    w_spline = 1，于是初始等价于 SiLU——与 KAN 论文的初始化一致。
    """

    def __init__(self, grid=8, degree=3, B=2.0, base_act="silu",
                 base_scale=1.0):
        super().__init__()
        self.grid = grid
        self.p = degree
        self.B = B
        self.base_scale = base_scale
        n_basis = grid + degree
        self.coeff = nn.Parameter(torch.zeros(n_basis))
        self.w_spline = nn.Parameter(torch.tensor(1.0))
        self.w_base = nn.Parameter(torch.tensor(1.0))
        h = (2 * B) / grid
        knots = (torch.arange(-degree, grid + degree + 1).float() * h - B)
        self.register_buffer("knots", knots)
        if base_act == "silu":
            self.base = nn.SiLU()
        elif base_act == "identity":
            self.base = nn.Identity()
        else:
            raise ValueError(base_act)

    def basis(self, x):
        """x: [...]; 返回 [..., n_basis]。向量化 Cox-de Boor 递推。"""
        t = self.knots
        shape = x.shape
        xf = x.reshape(-1, 1)                 # [N,1]
        p = self.p
        n = t.numel() - 1
        tk = t.reshape(1, -1)                 # [1, n+1]
        # 0 阶
        B = ((xf >= tk[:, :-1]) & (xf < tk[:, 1:])).to(x.dtype)   # [N,n]
        for j in range(1, p + 1):
            m = n - j                          # 新的基函数个数
            d1 = (tk[:, j:j + m] - tk[:, :m])                  # [1,m]
            d2 = (tk[:, j + 1:j + m + 1] - tk[:, 1:m + 1])     # [1,m]
            a = torch.where(d1 > 0, (xf - tk[:, :m]) / d1, torch.zeros(1, dtype=x.dtype)) * B[:, :m]
            b = torch.where(d2 > 0, (tk[:, j + 1:j + m + 1] - xf) / d2, torch.zeros(1, dtype=x.dtype)) * B[:, 1:m + 1]
            B = a + b
        return B.reshape(*shape, -1)

    def forward(self, z):
        x = z.clamp(-self.B, self.B)
        N = self.basis(x)                    # [..., n_basis]
        spline = torch.einsum("...j,j->...", N, self.coeff)
        return self.w_base * self.base(z) + self.w_spline * spline

    def extra_repr(self):
        return f"grid={self.grid}, p={self.p}, B={self.B}"


GATE_ACTS = {
    "swiglu": "silu", "geglu": "gelu", "polyglu": "poly",
}


# ============================================================
# 归一化层
# ============================================================

class L2UnitNorm(nn.Module):
    """射影归一化：y / ||y||_2，无额外可学习参数。"""

    def __init__(self, dim=None, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        norm = torch.linalg.vector_norm(x, ord=2, dim=1, keepdim=True)
        return x / norm.clamp_min(self.eps)


class RMSNorm1d(nn.Module):
    """RMSNorm：y / sqrt(mean(y^2) + eps) * g。"""

    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        rms = x.pow(2).mean(dim=1, keepdim=True).add(self.eps).sqrt()
        return x / rms * self.weight


def make_norm(name, dim):
    name = name.lower()
    if name == "none":
        return None
    if name == "l2":
        return L2UnitNorm(dim)
    if name == "batch":
        return nn.BatchNorm1d(dim)
    if name == "layer":
        return nn.LayerNorm(dim)
    if name == "rms":
        return RMSNorm1d(dim)
    raise ValueError(f"未知归一化: {name}")


# ============================================================
# 网络：depth 个线性层（depth-1 个隐藏层 + 1 个分类层）
# ============================================================

class Net(nn.Module):
    def __init__(self, depth=10, hidden=128, num_classes=10, activation="relu",
                 degree=3, lam=0.01, norm="none", norm_pos="post",
                 in_dim=784, lpoly_degree=6, bspline_grid=8):
        super().__init__()
        if depth < 2:
            raise ValueError("depth 至少为 2")
        if norm_pos not in ("pre", "post"):
            raise ValueError("norm_pos ∈ {pre, post}")

        self.depth = depth
        self.hidden = hidden
        self.activation_name = activation
        self.norm_name = norm
        self.norm_pos = norm_pos
        self.is_gated = activation in GATE_ACTS

        self.blocks = nn.ModuleList()
        self.norms = nn.ModuleList()

        if self.is_gated:
            width = hidden
            prev = in_dim
            for _ in range(depth - 1):
                self.blocks.append(nn.ModuleDict({
                    "value": nn.Linear(prev, width),
                    "gate": nn.Linear(prev, width),
                }))
                self.norms.append(make_norm(norm, width))
                prev = width
            gate_act_name = GATE_ACTS[activation]
            self.gate_act = make_activation(gate_act_name, degree=degree,
                                            lam=lam)
            self.gate_act_name = gate_act_name
        else:
            width = hidden
            prev = in_dim
            for _ in range(depth - 1):
                self.blocks.append(nn.ModuleDict({"lin": nn.Linear(prev, width)}))
                self.norms.append(make_norm(norm, width))
                prev = width
            self.act = make_activation(activation, degree=degree, lam=lam,
                                       lpoly_degree=lpoly_degree,
                                       bspline_grid=bspline_grid)

        self.width = width
        self.final = nn.Linear(width, num_classes)
        self._in_dims = [in_dim] + [width] * (depth - 2)
        self.reset_parameters()

    def reset_parameters(self):
        for block in self.blocks:
            for lin in block.values():
                nn.init.xavier_uniform_(lin.weight)
                nn.init.zeros_(lin.bias)
        nn.init.xavier_uniform_(self.final.weight)
        nn.init.zeros_(self.final.bias)
        for norm in self.norms:
            if isinstance(norm, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.ones_(norm.weight)
                nn.init.zeros_(norm.bias)
            elif isinstance(norm, RMSNorm1d):
                nn.init.ones_(norm.weight)

    def forward(self, x, return_features=False):
        x = x.reshape(x.shape[0], -1)
        features = []
        for i, block in enumerate(self.blocks):
            norm = self.norms[i]
            if self.is_gated:
                z = block["value"](x) * self.gate_act(block["gate"](x))
                x = norm(z) if norm is not None else z
            else:
                z = block["lin"](x)
                if norm is not None and self.norm_pos == "pre":
                    z = norm(z)
                x = self.act(z)
                if norm is not None and self.norm_pos == "post":
                    x = norm(x)
            features.append(x)
        logits = self.final(x)
        return (logits, features) if return_features else logits

    def n_params(self):
        return sum(p.numel() for p in self.parameters())

    def n_linear_layers(self):
        return self.depth

    def flops_per_sample(self):
        """前向 FLOPs（乘加各计一次）。门控层按两层线性 + 逐元素乘法计。"""
        total = 0
        prev = 784 if not self.is_gated else None
        widths = [self.width] * (self.depth - 1)
        ins = self._in_dims
        for i, w in enumerate(widths):
            k = 2 if self.is_gated else 1
            total += k * 2 * ins[i] * w
        total += 2 * ins[-1] * self.final.out_features
        return total


# ============================================================
# 自构数据集
# ============================================================

def make_synthetic(name, n=5000, seed=0, in_dim=2):
    rng = np.random.default_rng(seed)
    name = name.lower()

    if name == "rings":
        num_classes = 5
        y = rng.integers(0, num_classes, size=n)
        r = 0.25 + 0.32 * y + rng.normal(0, 0.045, size=n)
        th = rng.uniform(0, 2 * np.pi, size=n)
        X = np.stack([r * np.cos(th), r * np.sin(th)], 1)

    elif name == "blobs":
        num_classes = 5
        k = np.arange(num_classes)
        c = np.stack([np.cos(2 * np.pi * k / num_classes),
                      np.sin(2 * np.pi * k / num_classes)], 1)
        y = rng.integers(0, num_classes, size=n)
        X = c[y] + rng.normal(0, 0.22, size=(n, 2))

    elif name == "spiral":
        num_classes = 5
        per = n // num_classes
        X, y = [], []
        for c in range(num_classes):
            t = np.linspace(0.15, 1.0, per)
            th = 2 * np.pi * 2.2 * t + 2 * np.pi * c / num_classes
            r = 0.9 * t
            X.append(np.stack([r * np.cos(th), r * np.sin(th)], 1))
            y.append(np.full(per, c))
        X = np.concatenate(X) + rng.normal(0, 0.035, size=(n, 2))
        y = np.concatenate(y)
        n = len(y)

    elif name == "xor":
        num_classes = 2
        X = rng.uniform(-1, 1, size=(n, 2)) + rng.normal(0, 0.06, size=(n, 2))
        y = ((X[:, 0] * X[:, 1]) > 0).astype(np.int64)

    elif name == "checker":
        num_classes = 2
        X = rng.uniform(-1, 1, size=(n, 2))
        # 4x4 棋盘格
        ix = np.floor((X[:, 0] + 1) / 0.5).astype(int)
        iy = np.floor((X[:, 1] + 1) / 0.5).astype(int)
        X = X + rng.normal(0, 0.02, size=(n, 2))
        y = ((ix + iy) % 2).astype(np.int64)

    elif name == "cubes":
        # 3 维输入，3 阶交互（三路 XOR）
        num_classes = 2
        X = rng.uniform(0, 1, size=(n, 3))
        y = ((X[:, 0] > 0.5).astype(int) ^ (X[:, 1] > 0.5).astype(int)
             ^ (X[:, 2] > 0.5).astype(int)).astype(np.int64)

    elif name == "parity":
        # d 维二值特征，标签为奇偶校验（最高阶交互，线性不可分）
        num_classes = 2
        d = in_dim if in_dim > 2 else 10
        X = rng.integers(0, 2, size=(n, d)).astype(np.float32)
        y = (X.sum(1) % 2).astype(np.int64)
        in_dim = d

    elif name == "gauss_highdim":
        # 高维输入 + 低维流形：检验"高维输入射影归一化"论点
        num_classes = 5
        d = in_dim if in_dim > 2 else 50
        k = np.arange(num_classes)
        dirs = rng.normal(0, 1, size=(num_classes, 2))
        dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        y = rng.integers(0, num_classes, size=n)
        core = dirs[y] + rng.normal(0, 0.25, size=(n, 2))
        proj = rng.normal(0, 1, size=(2, d))
        X = core @ proj + rng.normal(0, 0.15, size=(n, d))
        in_dim = d

    else:
        raise ValueError(f"未知自构数据集: {name}")

    X = X.astype(np.float32)
    # 训练集统计量做标准化（不使用测试集信息）
    mu, sd = X.mean(0, keepdims=True), X.std(0, keepdims=True) + 1e-6
    X = (X - mu) / sd
    return X, y.astype(np.int64), num_classes, in_dim


def load_data(args):
    """返回 train/test 的 (X, y) 张量以及 num_classes / in_dim。"""
    if args.dataset == "mnist":
        tf = transforms.ToTensor()
        dl = bool(getattr(args, "allow_download", False))
        tr = datasets.MNIST(args.data_dir, train=True, download=dl,
                            transform=tf)
        te = datasets.MNIST(args.data_dir, train=False, download=dl,
                            transform=tf)
        Xtr = torch.stack([im for im, _ in tr]).squeeze(1)
        ytr = torch.tensor([lb for _, lb in tr])
        Xte = torch.stack([im for im, _ in te]).squeeze(1)
        yte = torch.tensor([lb for _, lb in te])
        # MNIST: 与原始脚本一致，只做 ToTensor，不做额外标准化
        n = Xtr.shape[1] * Xtr.shape[2]
        return (Xtr.reshape(len(Xtr), n), ytr, Xte.reshape(len(Xte), n), yte,
                10, n)

    if args.dataset in ("synth_train", "synth_test"):
        # 同一 generator 家族：train 用 synth_seed，test 用 synth_seed+10000
        seed_tr = args.synth_seed
        seed_te = args.synth_seed + 10_000
        name = args.synth_name
        Xtr, ytr, C, D = make_synthetic(name, args.synth_n, seed_tr, args.in_dim)
        Xte, yte, C2, D2 = make_synthetic(name, args.synth_n // 5, seed_te,
                                          args.in_dim)
        assert C == C2 and D == D2
        return (torch.from_numpy(Xtr), torch.from_numpy(ytr),
                torch.from_numpy(Xte), torch.from_numpy(yte), C, D)

    raise ValueError(f"未知数据集: {args.dataset}")


# ============================================================
# 训练 / 评估
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    loss_sum, correct, total = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        bs = yb.size(0)
        loss_sum += loss.item() * bs
        correct += (logits.argmax(1) == yb).sum().item()
        total += bs
    return loss_sum / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes=10, per_class=False):
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        pred = logits.argmax(1)
        bs = yb.size(0)
        loss_sum += loss.item() * bs
        correct += (pred == yb).sum().item()
        total += bs
        idx = yb * num_classes + pred
        confusion += torch.bincount(idx.cpu(),
                                    minlength=num_classes ** 2).reshape(
            num_classes, num_classes)
    out = {"loss": loss_sum / total, "accuracy": correct / total,
           "confusion": confusion.tolist()}
    if per_class:
        acc = confusion.diagonal().float() / confusion.sum(1).clamp_min(1).float()
        out["per_class"] = acc.tolist()
    return out


def diagnostics(model, loader, device):
    """固定 batch 上的逐层梯度范数与逐层激活范数（训练前后各测一次）。

    注意：这里必须开启梯度（不能加 @torch.no_grad()），否则 loss.backward()
    会报 "element 0 of tensors does not require grad"。
    """
    was_training = model.training
    model.train()
    xb, yb = next(iter(loader))
    xb, yb = xb.to(device), yb.to(device)
    criterion = nn.CrossEntropyLoss()
    model.zero_grad(set_to_none=True)
    logits, features = model(xb, return_features=True)
    loss = criterion(logits, yb)
    loss.backward()

    grad_per_lin = []
    for block in model.blocks:
        for lin in block.values():
            grad_per_lin.append(float(lin.weight.grad.norm().item()))
    grad_per_lin.append(float(model.final.weight.grad.norm().item()))

    act_mean = [float(torch.linalg.vector_norm(f, dim=1).mean().item())
                for f in features]
    act_std = [float(f.std().item()) for f in features]

    model.zero_grad(set_to_none=True)
    if not was_training:
        model.eval()
    return grad_per_lin, act_mean, act_std


# ============================================================
# 主流程
# ============================================================

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="mnist")
    p.add_argument("--synth-name", default="rings")
    p.add_argument("--synth-n", type=int, default=5000)
    p.add_argument("--synth-seed", type=int, default=0)
    p.add_argument("--in-dim", type=int, default=2)
    p.add_argument("--activation", default="relu")
    p.add_argument("--degree", type=int, default=3)
    p.add_argument("--lam", type=float, default=0.01)
    p.add_argument("--lpoly-degree", type=int, default=6)
    p.add_argument("--bspline-grid", type=int, default=8)
    p.add_argument("--norm", default="none")
    p.add_argument("--norm-pos", default="post", choices=["pre", "post"])
    p.add_argument("--depth", type=int, default=10)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data-dir", default="./data")
    # 本地实验全程离线（download=False），以保证复现不受网络影响。
    # 仓库不含 MNIST 原始文件，全新克隆时加这个开关让 torchvision 自动下载。
    p.add_argument("--allow-download", action="store_true",
                   help="MNIST 不在 --data-dir 时，允许 torchvision 自动下载")
    p.add_argument("--out-dir", default="./runs")
    p.add_argument("--tag", default="")
    p.add_argument("--legacy-loader", action="store_true",
                   help="不使用显式 generator（复现原脚本用）")
    p.add_argument("--no-diagnostics", action="store_true")
    args = p.parse_args()

    set_seed(args.seed)
    device = torch.device("cpu")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    Xtr, ytr, Xte, yte, num_classes, in_dim = load_data(args)
    train_ds = TensorDataset(Xtr, ytr)
    test_ds = TensorDataset(Xte, yte)

    if args.legacy_loader:
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  shuffle=True, num_workers=0)
    else:
        g = torch.Generator().manual_seed(args.seed)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                                  shuffle=True, num_workers=0, generator=g)
    eval_loader = DataLoader(train_ds, batch_size=1024, shuffle=False,
                             num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=1024, shuffle=False,
                             num_workers=0)
    diag_loader = DataLoader(train_ds, batch_size=512, shuffle=False,
                             num_workers=0)

    model = Net(depth=args.depth, hidden=args.hidden,
                num_classes=num_classes, activation=args.activation,
                degree=args.degree, lam=args.lam, norm=args.norm,
                norm_pos=args.norm_pos,
                in_dim=in_dim, lpoly_degree=args.lpoly_degree,
                bspline_grid=args.bspline_grid).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, args.epochs))

    if args.no_diagnostics:
        grad_init, act_init, actstd_init = [], [], []
    else:
        grad_init, act_init, actstd_init = diagnostics(model, diag_loader, device)

    history = {"train_loss": [], "train_acc": [], "test_loss": [],
               "test_acc": []}
    best_acc, best_epoch = -1.0, 0
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer,
                                          criterion, device)
        m = evaluate(model, test_loader, criterion, device, num_classes)
        scheduler.step()
        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["test_loss"].append(m["loss"])
        history["test_acc"].append(m["accuracy"])
        if m["accuracy"] > best_acc:
            best_acc, best_epoch = m["accuracy"], epoch
        print(f"[{args.dataset}/{args.synth_name}/"
              f"{args.activation}-d{args.degree}-lam{args.lam:g}/"
              f"norm={args.norm}/L{args.depth}/s{args.seed}] "
              f"ep{epoch:02d}/{args.epochs} "
              f"tr={tr_acc*100:.2f}% te={m['accuracy']*100:.2f}% "
              f"({time.time()-t0:.0f}s)", flush=True)

    seconds = time.time() - t0
    final = evaluate(model, test_loader, criterion, device, num_classes,
                     per_class=True)
    train_metrics = evaluate(model, eval_loader, criterion, device, num_classes)
    if args.no_diagnostics:
        grad_end, act_end, actstd_end = [], [], []
    else:
        grad_end, act_end, actstd_end = diagnostics(model, diag_loader, device)

    tag = f"{args.tag}_" if args.tag else ""
    cid = (f"{tag}{args.dataset}-{args.synth_name}-"
           f"{args.activation}-d{args.degree}-lam{args.lam:g}-{args.norm}"
           f"{'-pre' if args.norm_pos == 'pre' else ''}"
           f"-L{args.depth}-H{args.hidden}-"
           f"lr{args.lr:g}-e{args.epochs}-s{args.seed}")

    record = {
        "config_id": cid, "args": vars(args),
        "num_classes": num_classes, "in_dim": in_dim,
        "n_params": model.n_params(), "depth": args.depth,
        "flops_per_sample": model.flops_per_sample(),
        "test_accuracy": final["accuracy"], "test_loss": final["loss"],
        "train_accuracy": train_metrics["accuracy"],
        "train_loss": train_metrics["loss"],
        "per_class_accuracy": final["per_class"],
        "confusion": final["confusion"],
        "best_accuracy": best_acc, "best_epoch": best_epoch,
        "seconds": seconds, "seconds_per_epoch": seconds / args.epochs,
        "history": history,
        "grad_norm_init": grad_init, "grad_norm_end": grad_end,
        "act_norm_init": act_init, "act_norm_end": act_end,
        "act_std_init": actstd_init, "act_std_end": actstd_end,
    }
    with open(out_dir / f"{cid}.json", "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False)
    print(f"=== {cid}: test_acc={final['accuracy']*100:.2f}% "
          f"params={model.n_params()} time={seconds:.1f}s", flush=True)


if __name__ == "__main__":
    main()
