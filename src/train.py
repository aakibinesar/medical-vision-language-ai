import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader
from torchvision import models

from dataset import CXRDataset


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_model(name, n_labels):
    name = name.lower()
    if name == "densenet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        model.classifier = nn.Linear(model.classifier.in_features, n_labels)
    elif name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, n_labels)
    else:
        raise ValueError(f"Unknown model: {name}")
    return model


def compute_metrics(y_true, y_prob):
    out = {}
    try:
        out["auroc_macro"] = roc_auc_score(y_true, y_prob, average="macro")
        out["auprc_macro"] = average_precision_score(y_true, y_prob, average="macro")
    except ValueError:
        out["auroc_macro"] = float("nan")
        out["auprc_macro"] = float("nan")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", required=True)
    ap.add_argument("--val-csv", required=True)
    ap.add_argument("--img-root", required=True,
                     help="Root directory that image paths in the CSVs are relative to")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--model", default="resnet18", choices=["resnet18", "densenet121"])
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--label-columns", nargs="+", default=None,
                     help="Restrict to these CSV columns as labels (e.g. to ignore a report_text "
                          "column). Defaults to every non-path column.")
    args = ap.parse_args()
    set_seed(args.seed)

    train_ds = CXRDataset(args.train_csv, args.img_root, args.img_size, train=True,
                           label_columns=args.label_columns)
    val_ds = CXRDataset(args.val_csv, args.img_root, args.img_size, train=False,
                         label_columns=args.label_columns)
    n_labels = len(train_ds.label_columns)

    dl_train = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                           num_workers=args.num_workers, drop_last=True)
    dl_val = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                         num_workers=args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args.model, n_labels).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    history = []
    best_auroc = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        for xb, yb in dl_train:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_ds)

        model.eval()
        ys, ps = [], []
        with torch.no_grad():
            for xb, yb in dl_val:
                xb = xb.to(device)
                probs = torch.sigmoid(model(xb)).cpu().numpy()
                ys.append(yb.numpy())
                ps.append(probs)
        ys, ps = np.vstack(ys), np.vstack(ps)
        m = compute_metrics(ys, ps)
        print(f"Epoch {epoch}/{args.epochs}  train_loss={train_loss:.4f}  "
              f"val_auroc={m['auroc_macro']:.4f}  val_auprc={m['auprc_macro']:.4f}")
        history.append({"epoch": epoch, "train_loss": train_loss, **m})

        if m["auroc_macro"] > best_auroc:
            best_auroc = m["auroc_macro"]
            torch.save({
                "model_state": model.state_dict(),
                "model_name": args.model,
                "label_columns": train_ds.label_columns,
                "img_size": args.img_size,
            }, outdir / "best.pt")

    with open(outdir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"Best val AUROC: {best_auroc:.4f}. Checkpoint + history written to {outdir}/")


if __name__ == "__main__":
    main()
