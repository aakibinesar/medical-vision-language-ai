"""Generate a tiny synthetic image dataset with the same CSV/folder shape as
the real chest X-ray data, so the pipeline (train -> eval -> gradcam) can be
run and verified end-to-end in seconds, without downloading anything.

This is a code-correctness smoke test only. It proves the pipeline runs, not
that the model is any good — real training happens on the actual dataset
(locally or, given the 2GB GPU, on Kaggle/Colab).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image


def make_split(out_dir: Path, img_dir: Path, n_per_class, img_size, seed, prefix):
    rng = np.random.default_rng(seed)
    rows = []
    for label, name in [(0, "normal"), (1, "pneumonia")]:
        # give each class a distinct mean brightness so a real model *can*
        # learn something, rather than pure unlearnable noise
        base = 60 if label == 0 else 160
        for i in range(n_per_class):
            arr = np.clip(rng.normal(base, 25, size=(img_size, img_size, 3)), 0, 255).astype("uint8")
            fname = f"{prefix}_{name}_{i}.jpg"
            Image.fromarray(arr).save(img_dir / fname)
            rows.append((f"images/{fname}", label))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="data_smoke")
    ap.add_argument("--n-per-class", type=int, default=20)
    ap.add_argument("--img-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    train_rows = make_split(out_dir, img_dir, args.n_per_class, args.img_size, args.seed, "train")
    val_rows = make_split(out_dir, img_dir, max(2, args.n_per_class // 4), args.img_size, args.seed + 1, "val")
    test_rows = make_split(out_dir, img_dir, max(2, args.n_per_class // 4), args.img_size, args.seed + 2, "test")

    for name, rows in [("train.csv", train_rows), ("val.csv", val_rows), ("test.csv", test_rows)]:
        pd.DataFrame(rows, columns=["path", "Pneumonia"]).to_csv(out_dir / name, index=False)

    print(f"Synthetic smoke-test data written to {out_dir}/ "
          f"(train={len(train_rows)}, val={len(val_rows)}, test={len(test_rows)})")


if __name__ == "__main__":
    main()
