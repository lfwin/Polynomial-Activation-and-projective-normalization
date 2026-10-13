#!/usr/bin/env python3
"""诊断脚本：单独测 identity / bilinear 在 MNIST 10 层上的行为与耗时。"""
import sys, time, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch, torch.nn as nn
from harness import Net, load_data, train_one_epoch, evaluate, set_seed
from torch.utils.data import DataLoader, TensorDataset

class A: pass
a = A()
a.dataset = "mnist"
a.data_dir = str(Path(__file__).resolve().parent.parent / "data")
Xtr, ytr, Xte, yte, C, D = load_data(a)
tr = DataLoader(TensorDataset(Xtr, ytr), batch_size=128, shuffle=True, num_workers=0)
te = DataLoader(TensorDataset(Xte, yte), batch_size=1024, shuffle=False, num_workers=0)
print("data", Xtr.shape, Xte.shape, "classes", C)

crit = nn.CrossEntropyLoss()
for activation, depth in [("identity", 10), ("bilinear", 10), ("identity", 3),
                         ("polynomial_free", 10)]:
    if activation == "polynomial_free":
        continue
    set_seed(42)
    m = Net(depth=depth, hidden=128, num_classes=10, activation=activation,
            in_dim=D)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-4)
    t0 = time.time()
    n_steps = 0
    for ep in range(1, 4):
        trl, tra = train_one_epoch(m, tr, opt, crit, None)
        ev = evaluate(m, te, crit, None, 10)
        print(f"  {activation} L{depth} ep{ep}: train={tra*100:.2f}% "
              f"test={ev['accuracy']*100:.2f}% loss={ev['loss']:.4f} "
              f"({time.time()-t0:.0f}s)", flush=True)
    # 看超时是不是因为梯度爆炸
    gn = sum(float(p.grad.norm()**2) for p in m.parameters() if p.grad is not None) ** 0.5
    print(f"  {activation} L{depth} total grad norm = {gn:.3e}", flush=True)
