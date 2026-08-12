"""Turn the IU X-Ray / Open-i dataset (Kaggle mirror: raddar/chest-xrays-indiana-university)
into train/val/test CSVs with columns: path,report_text,Abnormal.

Source files expected:
  --reports-csv     indiana_reports.csv     (uid, MeSH, Problems, image, indication,
                                              comparison, findings, impression)
  --projections-csv indiana_projections.csv (uid, filename, projection)
  --img-root        images/images_normalized/ (PNG files named by `filename`)

Each `uid` is one study (effectively one patient in this corpus — there is no separate
patient ID field). Splitting by uid is therefore a patient-level split, unlike the
pneumonia dataset which had no patient IDs at all.

`Abnormal` is a WEAK label derived from the Problems/MeSH field: 1 unless it is exactly
"normal" (case-insensitive), matching a common convention for this dataset. This is a
heuristic report-derived label, not a clinician adjudication — call this out explicitly
in the model card and technical report per the leakage-audit requirement in the project
plan (doc 2, Sec 4.3: "label leakage and report-derived labels must be audited").
"""
import argparse
import random
from pathlib import Path

import pandas as pd


def build_report_text(row):
    parts = []
    for field in ("findings", "impression"):
        val = row.get(field)
        if isinstance(val, str) and val.strip():
            parts.append(val.strip())
    return " ".join(parts)


def derive_abnormal_label(problems):
    if not isinstance(problems, str) or not problems.strip():
        return None  # unlabeled — excluded
    return 0 if problems.strip().lower() == "normal" else 1


def split_by_uid(uids, val_frac, test_frac, seed):
    unique_uids = sorted(set(uids))
    rng = random.Random(seed)
    rng.shuffle(unique_uids)
    n = len(unique_uids)
    n_test = max(1, int(n * test_frac))
    n_val = max(1, int(n * val_frac))
    test_uids = set(unique_uids[:n_test])
    val_uids = set(unique_uids[n_test:n_test + n_val])
    train_uids = set(unique_uids[n_test + n_val:])
    return train_uids, val_uids, test_uids


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reports-csv", required=True)
    ap.add_argument("--projections-csv", required=True)
    ap.add_argument("--out-dir", default="data_iuxray")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    reports = pd.read_csv(args.reports_csv)
    projections = pd.read_csv(args.projections_csv)

    # One frontal image per study — avoids duplicate/lateral views inflating counts.
    frontal = projections[projections["projection"].str.lower() == "frontal"]
    frontal = frontal.drop_duplicates(subset="uid", keep="first")

    merged = frontal.merge(reports, on="uid", how="inner")
    merged["report_text"] = merged.apply(build_report_text, axis=1)
    merged["Abnormal"] = merged["Problems"].apply(derive_abnormal_label)

    before = len(merged)
    merged = merged[(merged["report_text"].str.len() > 0) & merged["Abnormal"].notna()]
    merged["Abnormal"] = merged["Abnormal"].astype(int)
    print(f"{before} frontal studies -> {len(merged)} with both a non-empty report and a "
          f"derivable Abnormal label ({before - len(merged)} dropped)")

    train_uids, val_uids, test_uids = split_by_uid(
        merged["uid"], args.val_frac, args.test_frac, args.seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, uid_set in [("train", train_uids), ("val", val_uids), ("test", test_uids)]:
        subset = merged[merged["uid"].isin(uid_set)]
        out = subset[["filename", "report_text", "Abnormal"]].rename(columns={"filename": "path"})
        out.to_csv(out_dir / f"{name}.csv", index=False)
        print(f"{name}.csv: {len(out)} studies "
              f"({out['Abnormal'].sum()} abnormal / {(out['Abnormal'] == 0).sum()} normal)")


if __name__ == "__main__":
    main()
