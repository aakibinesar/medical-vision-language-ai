"""Generate a tiny synthetic image+report-text paired dataset, matching the
IU X-Ray CSV shape (path,report_text,Abnormal), so retrieval_baseline.py and
text_robustness.py can be smoke-tested without downloading anything.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

TEMPLATES = {
    0: ["The lungs are clear. No abnormality is seen. Heart size is normal.",
        "Normal chest radiograph. No acute cardiopulmonary process.",
        "No focal consolidation, effusion, or pneumothorax. Normal study."],
    1: ["There is a focal opacity in the right lower lobe consistent with consolidation.",
        "Cardiomegaly is present with mild pulmonary vascular congestion.",
        "Bilateral pleural effusions are noted, greater on the right."],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data_multimodal_smoke")
    ap.add_argument("--n-per-class", type=int, default=8)
    ap.add_argument("--img-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out_dir)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for label, templates in TEMPLATES.items():
        base = 60 if label == 0 else 160
        for i in range(args.n_per_class):
            arr = np.clip(rng.normal(base, 25, size=(args.img_size, args.img_size, 3)), 0, 255).astype("uint8")
            fname = f"{'normal' if label == 0 else 'abnormal'}_{i}.jpg"
            Image.fromarray(arr).save(img_dir / fname)
            text = templates[i % len(templates)]
            rows.append((f"images/{fname}", text, label))

    df = pd.DataFrame(rows, columns=["path", "report_text", "Abnormal"])
    n = len(df)
    n_val = max(2, n // 6)
    n_test = max(2, n // 6)
    df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    test_df, val_df, train_df = df[:n_test], df[n_test:n_test + n_val], df[n_test + n_val:]
    for name, split in [("train", train_df), ("val", val_df), ("test", test_df)]:
        split.to_csv(out_dir / f"{name}.csv", index=False)

    print(f"Synthetic multimodal smoke data written to {out_dir}/ "
          f"(train={len(train_df)}, val={len(val_df)}, test={len(test_df)})")


if __name__ == "__main__":
    main()
