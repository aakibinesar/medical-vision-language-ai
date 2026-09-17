"""Combine multiple eval.py metrics.json files (one per training seed) into
a mean/std summary, for a real confidence-interval estimate on the headline
classification numbers instead of a single-seed point estimate.
"""
import argparse
import json
import statistics
from pathlib import Path


def mean_std(values):
    if len(values) == 1:
        return {"mean": values[0], "std": 0.0, "n": 1, "values": values}
    return {"mean": statistics.mean(values), "std": statistics.stdev(values),
            "n": len(values), "values": values}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-json", nargs="+", required=True,
                     help="Paths to metrics.json files from eval.py, one per seed")
    ap.add_argument("--seeds", nargs="+", type=int, required=True,
                     help="Seed label for each --metrics-json, same order")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()
    assert len(args.metrics_json) == len(args.seeds), "one seed label per metrics.json"

    per_seed = []
    for seed, path in zip(args.seeds, args.metrics_json):
        with open(path) as f:
            m = json.load(f)
        per_seed.append({"seed": seed, "auroc_macro": m["auroc_macro"], "auprc_macro": m["auprc_macro"],
                          "ece_before_calibration": m["ece_before_calibration"],
                          "ece_after_calibration": m["ece_after_calibration"]})

    summary = {
        "n_seeds": len(args.seeds),
        "seeds": args.seeds,
        "per_seed": per_seed,
        "auroc_macro": mean_std([p["auroc_macro"] for p in per_seed]),
        "auprc_macro": mean_std([p["auprc_macro"] for p in per_seed]),
        "ece_before_calibration": mean_std([p["ece_before_calibration"] for p in per_seed]),
        "ece_after_calibration": mean_std([p["ece_after_calibration"] for p in per_seed]),
        "note": "std from n<5 seeds is itself a noisy estimate - report as an honest range, "
                "not a precise confidence interval.",
    }
    print(json.dumps(summary, indent=2))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "metrics_seed_ci.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {outdir / 'metrics_seed_ci.json'}")


if __name__ == "__main__":
    main()
