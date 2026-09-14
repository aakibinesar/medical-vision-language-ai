"""Per-example error analysis: run a trained checkpoint on a test CSV that
carries report_text, classify each example as TP/TN/FP/FN at a threshold,
and write out the misclassified cases with their report text so patterns in
what the model gets wrong can actually be read, not guessed at.
"""
import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import CXRDataset
from train import build_model


@torch.no_grad()
def predict(model, dl, device):
    probs = []
    model.eval()
    for xb, _ in dl:
        probs.append(torch.sigmoid(model(xb.to(device))).cpu().numpy()[:, 0])
    import numpy as np
    return np.concatenate(probs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV with path,report_text,<label>")
    ap.add_argument("--img-root", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(ckpt["model_name"], len(ckpt["label_columns"]),
                         dropout_p=ckpt.get("dropout_p", 0.0)).to(device)
    model.load_state_dict(ckpt["model_state"])
    label_col = ckpt["label_columns"][0]

    ds = CXRDataset(args.csv, args.img_root, ckpt["img_size"], train=False, label_columns=[label_col])
    dl = DataLoader(ds, batch_size=16, shuffle=False, num_workers=2)
    probs = predict(model, dl, device)

    df = pd.read_csv(args.csv)
    df["pred_prob"] = probs
    df["pred_label"] = (df["pred_prob"] >= args.threshold).astype(int)
    df["outcome"] = "TN"
    df.loc[(df[label_col] == 1) & (df["pred_label"] == 1), "outcome"] = "TP"
    df.loc[(df[label_col] == 0) & (df["pred_label"] == 1), "outcome"] = "FP"
    df.loc[(df[label_col] == 1) & (df["pred_label"] == 0), "outcome"] = "FN"

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(outdir / "error_analysis_predictions.csv", index=False)

    counts = df["outcome"].value_counts().to_dict()
    print("Outcome counts:", counts)
    print(f"\nWrote {outdir / 'error_analysis_predictions.csv'} ({len(df)} rows)")

    for outcome in ("FP", "FN"):
        subset = df[df["outcome"] == outcome].sort_values(
            "pred_prob", ascending=(outcome == "FN"))
        print(f"\n--- {outcome} cases ({len(subset)} total) ---")
        for _, row in subset.head(10).iterrows():
            text = str(row["report_text"])[:200]
            print(f"[{row['pred_prob']:.3f}] {row['path']}: {text}")


if __name__ == "__main__":
    main()
