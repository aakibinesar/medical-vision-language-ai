"""Shortcut-learning probe (mandatory evaluation block in the project plan):
"Test whether patient, site, or device identity dominates representations."

Extracts frozen penultimate-layer features from the trained image classifier
on two datasets (e.g. IU X-Ray vs. Pneumonia — different hospitals, scanners,
patient populations) and fits a simple logistic-regression probe to predict
*which dataset an image came from* using only those features.

Interpretation: if the probe can trivially tell datasets apart (high
accuracy/AUROC), the model's representations are dominated by site/scanner-
specific cues rather than (or in addition to) pathology signal — a shortcut
risk. This doesn't by itself prove the classifier's actual predictions rely
on the shortcut, but a high score is a flag worth investigating further
(e.g. via the missing/noisy-text or Grad-CAM analyses); a near-chance score
is reassuring.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from dataset import CXRDataset
from train import build_model


def get_head_module(model, model_name):
    return model.classifier if model_name.lower() == "densenet121" else model.fc


@torch.no_grad()
def extract_features(model, head_module, dl, device):
    captured = {}

    def hook(module, inputs):
        captured["feat"] = inputs[0].detach().cpu().numpy()

    handle = head_module.register_forward_pre_hook(hook)
    feats = []
    for xb, _ in dl:
        model(xb.to(device))
        feats.append(captured["feat"])
    handle.remove()
    return np.vstack(feats)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--csv-a", required=True)
    ap.add_argument("--img-root-a", required=True)
    ap.add_argument("--label-column-a", required=True)
    ap.add_argument("--name-a", default="dataset_a")
    ap.add_argument("--csv-b", required=True)
    ap.add_argument("--img-root-b", required=True)
    ap.add_argument("--label-column-b", required=True)
    ap.add_argument("--name-b", default="dataset_b")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(ckpt["model_name"], len(ckpt["label_columns"]),
                         dropout_p=ckpt.get("dropout_p", 0.0)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    head = get_head_module(model, ckpt["model_name"])

    ds_a = CXRDataset(args.csv_a, args.img_root_a, ckpt["img_size"], train=False,
                       label_columns=[args.label_column_a])
    ds_b = CXRDataset(args.csv_b, args.img_root_b, ckpt["img_size"], train=False,
                       label_columns=[args.label_column_b])
    dl_a = DataLoader(ds_a, batch_size=args.batch_size, shuffle=False, num_workers=2)
    dl_b = DataLoader(ds_b, batch_size=args.batch_size, shuffle=False, num_workers=2)

    feats_a = extract_features(model, head, dl_a, device)
    feats_b = extract_features(model, head, dl_b, device)

    X = np.vstack([feats_a, feats_b])
    y = np.concatenate([np.zeros(len(feats_a)), np.ones(len(feats_b))])  # 0=A, 1=B

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=args.seed, stratify=y)
    probe = LogisticRegression(max_iter=2000)
    probe.fit(X_train, y_train)
    probe_probs = probe.predict_proba(X_test)[:, 1]
    probe_preds = probe.predict(X_test)

    # Majority-class baseline: what a probe gets "for free" from class imbalance alone.
    majority_acc = max(y_test.mean(), 1 - y_test.mean())

    result = {
        "dataset_a": args.name_a,
        "dataset_b": args.name_b,
        "n_a": len(feats_a),
        "n_b": len(feats_b),
        "probe_accuracy": float(accuracy_score(y_test, probe_preds)),
        "probe_auroc": float(roc_auc_score(y_test, probe_probs)),
        "majority_class_baseline_accuracy": float(majority_acc),
        "interpretation": (
            "probe_auroc near 0.5 = representations don't trivially separate datasets "
            "(reassuring). probe_auroc near 1.0 = the model's features strongly encode "
            "dataset-of-origin (site/scanner shortcut risk) - probe_accuracy should be read "
            "relative to majority_class_baseline_accuracy, not in isolation."
        ),
    }
    print(json.dumps(result, indent=2))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "shortcut_probe.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"Wrote {outdir / 'shortcut_probe.json'}")


if __name__ == "__main__":
    main()
