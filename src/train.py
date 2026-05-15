"""Training and evaluation loop."""
import argparse
from pathlib import Path

import torch
import torch.nn as nn

from models import LeNet, VGG11Small, get_resnet18, get_vgg16
from utils import get_dataloaders, plot_history

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESULTS_DIR = Path(__file__).parent.parent / "results" / "figures"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_REGISTRY = {
    "lenet": lambda: LeNet(num_classes=4),
    "vgg11small": lambda: VGG11Small(num_classes=4),
    "resnet18": lambda: get_resnet18(num_classes=4),
    "vgg16": lambda: get_vgg16(num_classes=4),
}


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss, total_acc = 0.0, 0.0
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_acc += (out.argmax(1) == y).float().mean().item()
    n = len(loader)
    return total_loss / n, total_acc / n


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss, total_acc = 0.0, 0.0
    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        out = model(X)
        total_loss += criterion(out, y).item()
        total_acc += (out.argmax(1) == y).float().mean().item()
    n = len(loader)
    return total_loss / n, total_acc / n


def train(model_name, data_dir, epochs=20, lr=1e-3, batch_size=32, img_size=64):
    train_dl, val_dl = get_dataloaders(data_dir, img_size=img_size, batch_size=batch_size)
    model = MODEL_REGISTRY[model_name]().to(DEVICE)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_dl, optimizer, criterion)
        vl_loss, vl_acc = evaluate(model, val_dl, criterion)
        history["train_loss"].append(tr_loss)
        history["val_loss"].append(vl_loss)
        history["train_acc"].append(tr_acc)
        history["val_acc"].append(vl_acc)
        print(f"[{epoch:03d}/{epochs}] loss {tr_loss:.4f}/{vl_loss:.4f}  acc {tr_acc:.4f}/{vl_acc:.4f}")

    plot_history(history, title=model_name, save_path=RESULTS_DIR / f"{model_name}_history.png")
    torch.save(model.state_dict(), RESULTS_DIR.parent / f"{model_name}.pth")
    return model, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="lenet", choices=list(MODEL_REGISTRY))
    parser.add_argument("--data", default="./data/dataset2-master/dataset2-master/images")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--img_size", type=int, default=64)
    args = parser.parse_args()
    train(args.model, args.data, args.epochs, args.lr, args.batch_size, args.img_size)
