
import argparse
import random
import json
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# 三阶多项式激活 + 射影归一化
# f(z) = z + lam*z^3
# pi(f(z)) = f(z) / ||f(z)||_2
# ============================================================

class ProjectivePolynomialActivation(nn.Module):
    def __init__(self, lam=0.01, eps=1e-8):
        super().__init__()
        self.lam = lam
        self.eps = eps

    def forward(self, z):
        z = z.clamp(-10.0, 10.0)
        y = z + self.lam * z.pow(3)

        norm = torch.linalg.vector_norm(
            y, ord=2, dim=1, keepdim=True
        )
        return y / norm.clamp_min(self.eps)


# ============================================================
# 10 层网络
# hidden_dim 可以设置为 128 或 256
# num_layers 表示总线性层数，包括最后的分类层
# ============================================================

class ProjectivePolynomialNet(nn.Module):
    def __init__(
        self,
        hidden_dim=128,
        num_layers=10,
        num_classes=10,
        lam=0.01,
    ):
        super().__init__()

        if num_layers < 2:
            raise ValueError("num_layers 必须至少为 2")

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # 第 1 个线性层：MNIST 输入 784 -> hidden_dim
        self.layers = nn.ModuleList()
        self.activations = nn.ModuleList()

        self.layers.append(nn.Linear(784, hidden_dim))

        # 前 num_layers - 1 层均为隐藏层
        for _ in range(num_layers - 2):
            self.layers.append(
                nn.Linear(hidden_dim, hidden_dim)
            )

        # 最后一层输出 10 类 logits
        self.layers.append(
            nn.Linear(hidden_dim, num_classes)
        )

        # 每个隐藏层都有多项式激活和射影归一化
        for _ in range(num_layers - 1):
            self.activations.append(
                ProjectivePolynomialActivation(lam=lam)
            )

        for layer in self.layers:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x, return_features=False):
        x = x.reshape(x.shape[0], -1)
        features = []

        # 所有隐藏层：Linear -> Polynomial -> Projection
        for i in range(self.num_layers - 1):
            x = self.layers[i](x)
            x = self.activations[i](x)
            features.append(x)

        # 最后一层：Linear -> logits，不做激活
        logits = self.layers[-1](x)

        if return_features:
            return logits, features

        return logits


# ============================================================
# 训练
# ============================================================

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    loss_sum = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()

        batch_size = labels.size(0)
        loss_sum += loss.item() * batch_size
        correct += (logits.argmax(1) == labels).sum().item()
        total += batch_size

    return loss_sum / total, correct / total


# ============================================================
# 测试评估
# ============================================================

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    loss_sum = 0.0
    correct = 0
    total = 0
    confusion = torch.zeros(10, 10, dtype=torch.int64)

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)
        pred = logits.argmax(1)

        batch_size = labels.size(0)
        loss_sum += loss.item() * batch_size
        correct += (pred == labels).sum().item()
        total += batch_size

        idx = labels * 10 + pred
        confusion += torch.bincount(
            idx.cpu(), minlength=100
        ).reshape(10, 10)

    return {
        "loss": loss_sum / total,
        "accuracy": correct / total,
        "confusion": confusion.numpy(),
    }


# ============================================================
# 可视化 1：训练和测试曲线
# ============================================================

def plot_curves(history, out_dir):
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["test_loss"], label="Test")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross Entropy")
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    axes[1].plot(
        epochs, np.array(history["train_acc"]) * 100,
        label="Train"
    )
    axes[1].plot(
        epochs, np.array(history["test_acc"]) * 100,
        label="Test"
    )
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].grid(alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(out_dir / "training_curves.png", dpi=180)
    plt.close(fig)


# ============================================================
# 可视化 2：混淆矩阵
# ============================================================

def plot_confusion(confusion, out_dir):
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(confusion, cmap="Blues")
    ax.set_title("MNIST Test Confusion Matrix")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))

    threshold = confusion.max() * 0.55
    for i in range(10):
        for j in range(10):
            ax.text(
                j, i, str(confusion[i, j]),
                ha="center", va="center",
                color="white" if confusion[i, j] > threshold else "black",
                fontsize=8,
            )

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=180)
    plt.close(fig)


# ============================================================
# 可视化 3：各类别准确率
# ============================================================

def plot_class_accuracy(confusion, out_dir):
    accuracy = np.diag(confusion) / np.maximum(
        confusion.sum(axis=1), 1
    )

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(range(10), accuracy * 100)
    ax.set_xticks(range(10))
    ax.set_ylim(0, 105)
    ax.set_xlabel("Digit")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Per-Class Accuracy")
    ax.grid(axis="y", alpha=0.3)

    for i, value in enumerate(accuracy):
        ax.text(i, value * 100 + 1, f"{value*100:.1f}%",
                ha="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_dir / "per_class_accuracy.png", dpi=180)
    plt.close(fig)

    return accuracy


# ============================================================
# 可视化 4：测试集预测样本
# ============================================================

@torch.no_grad()
def plot_predictions(model, dataset, device, out_dir, count=25):
    model.eval()
    rng = np.random.default_rng(42)
    ids = rng.choice(len(dataset), size=count, replace=False)

    images = torch.stack([dataset[int(i)][0] for i in ids])
    labels = torch.tensor([dataset[int(i)][1] for i in ids])

    logits = model(images.to(device))
    probs = logits.softmax(dim=1)
    confidence, pred = probs.max(dim=1)

    fig, axes = plt.subplots(5, 5, figsize=(11, 12))

    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i, 0].numpy(), cmap="gray")
        true_label = int(labels[i])
        predicted_label = int(pred[i].item())

        ax.set_title(
            f"True={true_label}, Pred={predicted_label}\n"
            f"Conf={confidence[i].item():.2f}",
            color="green" if true_label == predicted_label else "red",
            fontsize=9,
        )
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(out_dir / "sample_predictions.png", dpi=180)
    plt.close(fig)


# ============================================================
# 可视化 5：所有隐藏层射影化之后的范数
# ============================================================

@torch.no_grad()
def plot_hidden_norms(model, loader, device, out_dir):
    model.eval()
    layer_norms = [[] for _ in range(model.num_layers - 1)]

    for batch_idx, (images, _) in enumerate(loader):
        if batch_idx >= 10:
            break

        _, features = model(
            images.to(device), return_features=True
        )

        for i, h in enumerate(features):
            norms = torch.linalg.vector_norm(h, dim=1)
            layer_norms[i].extend(norms.cpu().tolist())

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, norms in enumerate(layer_norms):
        ax.hist(
            norms, bins=30, alpha=0.35,
            label=f"Hidden layer {i+1}",
        )

    ax.set_title("Projective Activation Norms")
    ax.set_xlabel("L2 Norm")
    ax.set_ylabel("Count")
    ax.legend(ncol=3, fontsize=8)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_dir / "projective_norms.png", dpi=180)
    plt.close(fig)


# ============================================================
# 主程序
# ============================================================

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)

    # 支持 128 或 256，也允许传入其他正整数
    parser.add_argument("--hidden-dim", type=int, default=128)

    # 总线性层数：10 表示 9 个隐藏层 + 1 个分类层
    parser.add_argument("--num-layers", type=int, default=10)

    parser.add_argument("--lam", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--output-dir", default="./results_10layers")

    args = parser.parse_args()

    if args.hidden_dim < 1:
        raise ValueError("hidden-dim 必须为正整数")
    if args.num_layers < 2:
        raise ValueError("num-layers 必须至少为 2")
    if args.epochs < 1:
        raise ValueError("epochs 必须至少为 1")
    if args.lam < 0:
        raise ValueError("lam 必须大于或等于 0")

    set_seed(args.seed)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print(f"Device: {device}")
    print(f"Linear layers: {args.num_layers}")
    print(f"Hidden neurons per layer: {args.hidden_dim}")
    print(f"Hidden projective activations: {args.num_layers - 1}")

    transform = transforms.ToTensor()

    train_dataset = datasets.MNIST(
        args.data_dir, train=True, download=True,
        transform=transform,
    )
    test_dataset = datasets.MNIST(
        args.data_dir, train=False, download=True,
        transform=transform,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = ProjectivePolynomialNet(
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        lam=args.lam,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs
    )

    history = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
    }

    best_accuracy = -1.0
    best_path = out_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, device
        )

        metrics = evaluate(
            model, test_loader, criterion, device
        )
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(metrics["loss"])
        history["test_acc"].append(metrics["accuracy"])

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"Train Loss={train_loss:.4f} | "
            f"Train Acc={train_acc*100:.2f}% | "
            f"Test Loss={metrics['loss']:.4f} | "
            f"Test Acc={metrics['accuracy']*100:.2f}%"
        )

        if metrics["accuracy"] > best_accuracy:
            best_accuracy = metrics["accuracy"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": vars(args),
                    "epoch": epoch,
                    "test_accuracy": best_accuracy,
                },
                best_path,
            )

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    final_metrics = evaluate(
        model, test_loader, criterion, device
    )
    confusion = final_metrics["confusion"]

    plot_curves(history, out_dir)
    plot_confusion(confusion, out_dir)
    class_accuracy = plot_class_accuracy(confusion, out_dir)
    plot_predictions(model, test_dataset, device, out_dir)
    plot_hidden_norms(model, test_loader, device, out_dir)

    results = {
        "best_epoch": int(checkpoint["epoch"]),
        "hidden_dim": args.hidden_dim,
        "num_layers": args.num_layers,
        "test_loss": float(final_metrics["loss"]),
        "test_accuracy": float(final_metrics["accuracy"]),
        "test_accuracy_percent": float(
            final_metrics["accuracy"] * 100
        ),
        "per_class_accuracy": {
            str(i): float(class_accuracy[i]) for i in range(10)
        },
        "config": vars(args),
    }

    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print("\n========== FINAL TEST RESULTS ==========")
    print(f"Best epoch: {checkpoint['epoch']}")
    print(f"Test loss: {final_metrics['loss']:.4f}")
    print(f"Test accuracy: {final_metrics['accuracy']*100:.2f}%")

    for digit in range(10):
        print(
            f"Digit {digit}: {class_accuracy[digit]*100:.2f}%"
        )

    print(f"Results directory: {out_dir.resolve()}")


if __name__ == "__main__":
    main()


