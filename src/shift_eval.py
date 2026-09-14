"""Cross-dataset distribution-shift test (mandatory evaluation block in the
project plan). Runs a checkpoint trained on one dataset (IU X-Ray) against a
different dataset (Pneumonia — different hospital, patient population, and
acquisition setting, no report text) to see how much performance and
calibration degrade outside the training distribution.

Note the label semantics differ: IU X-Ray's `Abnormal` covers ANY abnormal
finding, while Pneumonia's `Pneumonia` label is specific to one condition.
This is not an apples-to-apples accuracy comparison — a model trained to
detect "any abnormality" is not expected to specifically detect pneumonia.
The honest use of this script is the *shape* of degradation (AUROC drop,
calibration drift) as a shift-robustness signal, not a claim that the two
labels mean the same thing. Report this limitation alongside any numbers.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader

from dataset import CXRDataset
from eval import expected_calibration_error
from train import build_model


@torch.no_grad()
def predict(model, dl, device):
    ys, probs = [], []
    model.eval()
    for xb, yb in dl:
        xb = xb.to(device)
        p = torch.sigmoid(model(xb)).cpu().numpy()
        probs.append(p)
        ys.append(yb.numpy())
    return np.vstack(ys), np.vstack(probs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True, help="Checkpoint trained on the source dataset")
    ap.add_argument("--shift-csv", required=True, help="Target-dataset CSV, e.g. pneumonia test.csv")
    ap.add_argument("--shift-img-root", required=True)
    ap.add_argument("--shift-target-column", required=True,
                     help="Which column in --shift-csv is ground truth (e.g. 'Pneumonia')")
    ap.add_argument("--in-distribution-metrics", default=None,
                     help="Optional path to the source dataset's results/metrics.json, "
                          "to print a side-by-side comparison")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(ckpt["model_name"], len(ckpt["label_columns"]),
                         dropout_p=ckpt.get("dropout_p", 0.0)).to(device)
    model.load_state_dict(ckpt["model_state"])

    ds = CXRDataset(args.shift_csv, args.shift_img_root, ckpt["img_size"], train=False,
                     label_columns=[args.shift_target_column])
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    y_true, probs = predict(model, dl, device)
    y_true, probs = y_true[:, 0], probs[:, 0]

    shift_metrics = {
        "source_label": ckpt["label_columns"][0],
        "target_label": args.shift_target_column,
        "n_examples": len(y_true),
        "auroc": roc_auc_score(y_true, probs) if len(set(y_true)) > 1 else float("nan"),
        "auprc": average_precision_score(y_true, probs) if len(set(y_true)) > 1 else float("nan"),
        "ece": expected_calibration_error(y_true, probs),
        "mean_predicted_prob": float(probs.mean()),
    }
    print("Shift-target metrics:")
    print(json.dumps(shift_metrics, indent=2))

    if args.in_distribution_metrics:
        with open(args.in_distribution_metrics) as f:
            id_metrics = json.load(f)
        print("\nIn-distribution (source dataset test set) metrics for comparison:")
        print(json.dumps({k: id_metrics.get(k) for k in
                           ("auroc_macro", "auprc_macro", "ece_after_calibration")}, indent=2))
        shift_metrics["in_distribution_comparison"] = {
            k: id_metrics.get(k) for k in ("auroc_macro", "auprc_macro", "ece_after_calibration")
        }

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "shift_metrics.json", "w") as f:
        json.dump(shift_metrics, f, indent=2)
    print(f"\nWrote {outdir / 'shift_metrics.json'}")


if __name__ == "__main__":
    main()
