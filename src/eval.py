import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, confusion_matrix, roc_auc_score, roc_curve
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import CXRDataset
from train import build_model


@torch.no_grad()
def collect_logits(model, dl, device):
    ys, logits = [], []
    model.eval()
    for xb, yb in dl:
        xb = xb.to(device)
        logits.append(model(xb).cpu().numpy())
        ys.append(yb.numpy())
    return np.vstack(ys), np.vstack(logits)


def fit_temperature(logits, targets, device):
    """Single-parameter temperature scaling (Guo et al. 2017), fit on the val set."""
    logits_t = torch.tensor(logits, dtype=torch.float32, device=device)
    targets_t = torch.tensor(targets, dtype=torch.float32, device=device)
    temperature = torch.nn.Parameter(torch.ones(1, device=device))
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.LBFGS([temperature], lr=0.05, max_iter=100)

    def closure():
        optimizer.zero_grad()
        loss = criterion(logits_t / temperature, targets_t)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(temperature.detach().cpu().item())


def expected_calibration_error(y_true, y_prob, n_bins=10):
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (y_prob >= lo) & (y_prob < hi) if hi < 1 else (y_prob >= lo) & (y_prob <= hi)
        if mask.sum() == 0:
            continue
        conf = y_prob[mask].mean()
        acc = y_true[mask].mean()
        ece += (mask.sum() / n) * abs(acc - conf)
    return float(ece)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-csv", required=True, help="Used to fit temperature scaling")
    ap.add_argument("--test-csv", required=True, help="Held-out set the metrics are reported on")
    ap.add_argument("--img-root", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(ckpt["model_name"], len(ckpt["label_columns"]),
                         dropout_p=ckpt.get("dropout_p", 0.0)).to(device)
    model.load_state_dict(ckpt["model_state"])
    img_size = ckpt["img_size"]
    label_columns = ckpt["label_columns"]

    val_ds = CXRDataset(args.val_csv, args.img_root, img_size, train=False, label_columns=label_columns)
    test_ds = CXRDataset(args.test_csv, args.img_root, img_size, train=False, label_columns=label_columns)
    dl_val = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    dl_test = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    y_val, logits_val = collect_logits(model, dl_val, device)
    y_test, logits_test = collect_logits(model, dl_test, device)

    probs_test_raw = 1 / (1 + np.exp(-logits_test))

    # Calibration: fit temperature on val, apply to test (single label -> flatten)
    temperature = fit_temperature(logits_val[:, :1], y_val[:, :1], device)
    probs_test_calibrated = 1 / (1 + np.exp(-logits_test / temperature))

    metrics = {
        "auroc_macro": roc_auc_score(y_test, probs_test_raw, average="macro"),
        "auprc_macro": average_precision_score(y_test, probs_test_raw, average="macro"),
        "temperature": temperature,
        "ece_before_calibration": expected_calibration_error(y_test[:, 0], probs_test_raw[:, 0]),
        "ece_after_calibration": expected_calibration_error(y_test[:, 0], probs_test_calibrated[:, 0]),
    }
    per_label = {}
    for i, name in enumerate(label_columns):
        per_label[name] = {
            "auroc": roc_auc_score(y_test[:, i], probs_test_raw[:, i]),
            "auprc": average_precision_score(y_test[:, i], probs_test_raw[:, i]),
        }
    metrics["per_label"] = per_label

    with open(outdir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))

    # Confusion matrix (threshold 0.5) for the primary label
    y_pred = (probs_test_raw[:, 0] >= 0.5).astype(int)
    cm = confusion_matrix(y_test[:, 0], y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, str(v), ha="center", va="center")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    class_labels = ["Normal", label_columns[0]]
    ax.set_xticklabels(class_labels); ax.set_yticklabels(class_labels)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix ({label_columns[0]})")
    fig.tight_layout()
    fig.savefig(outdir / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test[:, 0], probs_test_raw[:, 0])
    fig, ax = plt.subplots(figsize=(4, 4))
    ax.plot(fpr, tpr, label=f"AUROC = {metrics['auroc_macro']:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("ROC curve"); ax.legend()
    fig.tight_layout()
    fig.savefig(outdir / "roc_curve.png", dpi=150)
    plt.close(fig)

    print(f"Wrote metrics.json, confusion_matrix.png, roc_curve.png to {outdir}/")


if __name__ == "__main__":
    main()
