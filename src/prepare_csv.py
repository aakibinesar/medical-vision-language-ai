"""Turn the Kaggle chest_xray/{train,test,val}/{NORMAL,PNEUMONIA} folder tree
into train.csv / val.csv / test.csv with columns: path,Pneumonia (0/1).

The dataset's shipped val/ split only has 16 images, so we pool train+val
and make our own stratified split; test/ is left untouched as the held-out set.
"""
import argparse
import random
from pathlib import Path

import pandas as pd

CLASSES = {"NORMAL": 0, "PNEUMONIA": 1}


def list_images(split_dir: Path):
    rows = []
    for cls_name, label in CLASSES.items():
        cls_dir = split_dir / cls_name
        if not cls_dir.is_dir():
            continue
        for p in sorted(cls_dir.glob("*")):
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                rows.append((str(p.relative_to(split_dir.parent.parent)), label))
    return rows


def stratified_split(rows, val_frac, seed):
    by_label = {0: [], 1: []}
    for r in rows:
        by_label[r[1]].append(r)
    rng = random.Random(seed)
    train_rows, val_rows = [], []
    for label, items in by_label.items():
        rng.shuffle(items)
        n_val = max(1, int(len(items) * val_frac))
        val_rows.extend(items[:n_val])
        train_rows.extend(items[n_val:])
    rng.shuffle(train_rows)
    rng.shuffle(val_rows)
    return train_rows, val_rows


def write_csv(rows, out_path):
    df = pd.DataFrame(rows, columns=["path", "Pneumonia"])
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(df)} images, "
          f"{df['Pneumonia'].sum()} pneumonia / {(df['Pneumonia'] == 0).sum()} normal)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True,
                     help="Path to the extracted chest_xray/ folder (contains train/ test/ val/)")
    ap.add_argument("--out-dir", default="data")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data_root = Path(args.data_root)
    pooled = list_images(data_root / "train") + list_images(data_root / "val")
    test_rows = list_images(data_root / "test")

    train_rows, val_rows = stratified_split(pooled, args.val_frac, args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(train_rows, out_dir / "train.csv")
    write_csv(val_rows, out_dir / "val.csv")
    write_csv(test_rows, out_dir / "test.csv")


if __name__ == "__main__":
    main()
