"""Simple image-text fusion baseline (the last item in the project plan's
"Minimum Model Stack") — and, more importantly, the actual Gate 2 test:
"Text or VLM component adds performance, calibration, robustness, or
interpretability - not merely complexity."

Uses frozen pretrained BiomedCLIP encoders (no fine-tuning, avoiding the
"compute overload" risk the project plan explicitly flags) to extract image
and text embeddings, then trains a lightweight logistic-regression head on
three feature sets from the *same* train/val/test split:
  - image-only  (BiomedCLIP image embedding)
  - text-only   (BiomedCLIP text embedding)
  - fusion      (concatenation of both)

Comparing all three on identical data is what actually answers the Gate 2
question - not just "did we build a fusion model" but "did fusion help".
A regularization strength is selected per feature set via the val split
(small grid), and calibration (ECE) is reported the same way as eval.py's
image-only CNN baseline, so all baselines in this project are comparable.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

from clip_utils import DEFAULT_MODEL_ID, embed_images, embed_texts, load_clip, load_pairs
from eval import expected_calibration_error, fit_temperature


def fit_and_eval(X_train, y_train, X_val, y_val, X_test, y_test, device):
    best_C, best_val_auroc = 1.0, -1.0
    for C in (0.01, 0.1, 1.0, 10.0):
        clf = LogisticRegression(max_iter=2000, C=C)
        clf.fit(X_train, y_train)
        val_probs = clf.predict_proba(X_val)[:, 1]
        auroc = roc_auc_score(y_val, val_probs) if len(set(y_val)) > 1 else 0.0
        if auroc > best_val_auroc:
            best_val_auroc, best_C = auroc, C

    clf = LogisticRegression(max_iter=2000, C=best_C)
    clf.fit(X_train, y_train)

    val_logits = clf.decision_function(X_val).reshape(-1, 1)
    test_logits = clf.decision_function(X_test).reshape(-1, 1)
    test_probs_raw = clf.predict_proba(X_test)[:, 1]

    temperature = fit_temperature(val_logits, y_val.reshape(-1, 1).astype(np.float32), device)
    test_probs_cal = 1 / (1 + np.exp(-test_logits.flatten() / temperature))

    return {
        "best_C": best_C,
        "test_auroc": roc_auc_score(y_test, test_probs_raw) if len(set(y_test)) > 1 else float("nan"),
        "test_auprc": average_precision_score(y_test, test_probs_raw) if len(set(y_test)) > 1 else float("nan"),
        "ece_before_calibration": expected_calibration_error(y_test, test_probs_raw),
        "ece_after_calibration": expected_calibration_error(y_test, test_probs_cal),
        "temperature": temperature,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", required=True, help="CSV with path,report_text,<label>")
    ap.add_argument("--val-csv", required=True)
    ap.add_argument("--test-csv", required=True)
    ap.add_argument("--img-root", required=True)
    ap.add_argument("--label-column", required=True)
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess, tokenizer = load_clip(args.model_id, device)

    splits = {}
    for name, csv_path in [("train", args.train_csv), ("val", args.val_csv), ("test", args.test_csv)]:
        df = load_pairs(csv_path)
        img_emb = embed_images(model, preprocess, args.img_root, df["path"].tolist(), device, args.batch_size).numpy()
        txt_emb = embed_texts(model, tokenizer, df["report_text"].tolist(), device, args.batch_size).numpy()
        y = df[args.label_column].values.astype(int)
        splits[name] = {"image": img_emb, "text": txt_emb, "fusion": np.concatenate([img_emb, txt_emb], axis=1), "y": y}
        print(f"{name}: {len(df)} examples, {y.sum()} positive / {(y == 0).sum()} negative")

    results = {}
    for feature_set in ("image", "text", "fusion"):
        results[feature_set] = fit_and_eval(
            splits["train"][feature_set], splits["train"]["y"],
            splits["val"][feature_set], splits["val"]["y"],
            splits["test"][feature_set], splits["test"]["y"],
            device,
        )
        print(f"\n{feature_set}-only feature set:" if feature_set != "fusion" else "\nfusion (image+text):")
        print(json.dumps(results[feature_set], indent=2))

    fusion_auroc = results["fusion"]["test_auroc"]
    best_unimodal_auroc = max(results["image"]["test_auroc"], results["text"]["test_auroc"])
    results["gate_2_verdict"] = {
        "fusion_beats_best_unimodal_auroc": bool(fusion_auroc > best_unimodal_auroc),
        "fusion_auroc": fusion_auroc,
        "best_unimodal_auroc": best_unimodal_auroc,
        "note": "Per the project plan, Gate 2 requires fusion to add measurable value "
                "(performance, calibration, robustness, or interpretability) beyond the "
                "strongest unimodal baseline - not just added complexity.",
    }
    print("\nGate 2 verdict:")
    print(json.dumps(results["gate_2_verdict"], indent=2))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "fusion_baseline.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {outdir / 'fusion_baseline.json'}")


if __name__ == "__main__":
    main()
