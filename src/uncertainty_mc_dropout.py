"""Monte Carlo dropout uncertainty estimation (Gal & Ghahramani 2016).

Requires a checkpoint trained with --dropout-p > 0 (train.py). Runs T
stochastic forward passes per image with dropout active, and reports the
predictive mean (used as the point prediction) and the standard deviation
across passes (an epistemic-uncertainty proxy) per example. High-uncertainty
cases are exactly the ones a safety-conscious system should flag for human
review rather than auto-predicting on — see abstention_eval.py, which
consumes this script's output.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from dataset import CXRDataset
from train import build_model


def enable_mc_dropout(model):
    """Set only Dropout layers to train() mode; everything else (BatchNorm
    included) stays in eval() mode so single-image inference remains stable."""
    model.eval()
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()


@torch.no_grad()
def mc_dropout_predict(model, dl, device, n_samples):
    """Returns (y_true, mean_probs, std_probs), each shape (N, n_labels)."""
    all_ys = []
    per_pass_probs = []  # list of (N, n_labels) arrays, one per MC sample
    for t in range(n_samples):
        ys, ps = [], []
        for xb, yb in dl:
            xb = xb.to(device)
            probs = torch.sigmoid(model(xb)).cpu().numpy()
            ps.append(probs)
            if t == 0:
                ys.append(yb.numpy())
        per_pass_probs.append(np.vstack(ps))
        if t == 0:
            all_ys = np.vstack(ys)
    stacked = np.stack(per_pass_probs, axis=0)  # (T, N, n_labels)
    return all_ys, stacked.mean(axis=0), stacked.std(axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--img-root", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n-samples", type=int, default=20, help="Number of MC-dropout forward passes")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    if ckpt.get("dropout_p", 0.0) <= 0:
        raise SystemExit(
            "This checkpoint was trained with dropout_p=0 — MC-dropout needs a checkpoint "
            "trained with --dropout-p > 0 (retrain with e.g. --dropout-p 0.3)."
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(ckpt["model_name"], len(ckpt["label_columns"]),
                         dropout_p=ckpt["dropout_p"]).to(device)
    model.load_state_dict(ckpt["model_state"])
    enable_mc_dropout(model)

    ds = CXRDataset(args.csv, args.img_root, ckpt["img_size"], train=False,
                     label_columns=ckpt["label_columns"])
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    y_true, mean_probs, std_probs = mc_dropout_predict(model, dl, device, args.n_samples)

    label_name = ckpt["label_columns"][0]
    out_df = pd.DataFrame({
        "path": ds.paths,
        "y_true": y_true[:, 0].astype(int),
        "mean_prob": mean_probs[:, 0],
        "std_prob": std_probs[:, 0],
    })
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    out_path = outdir / "mc_dropout_predictions.csv"
    out_df.to_csv(out_path, index=False)

    summary = {
        "label": label_name,
        "n_samples": args.n_samples,
        "n_examples": len(out_df),
        "mean_predictive_std": float(std_probs[:, 0].mean()),
        "std_correct_vs_incorrect": {
            "correct_pred_mean_std": float(out_df.loc[
                (out_df.mean_prob >= 0.5).astype(int) == out_df.y_true, "std_prob"].mean()),
            "incorrect_pred_mean_std": float(out_df.loc[
                (out_df.mean_prob >= 0.5).astype(int) != out_df.y_true, "std_prob"].mean()),
        },
    }
    with open(outdir / "mc_dropout_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    print(f"Wrote {out_path} and mc_dropout_summary.json to {outdir}/")


if __name__ == "__main__":
    main()
