"""Combine repeated-seed Gate 3 diagnostic results (shift_eval.py,
shortcut_probe.py, uncertainty_mc_dropout.py + abstention_eval.py) into
mean/std summaries, extending the same repeated-seed-CI treatment
aggregate_seed_metrics.py gave the Gate 1 classification headline numbers.
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
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--results-dirs", nargs="+", required=True,
                     help="One results dir per seed, same order as --seeds, each containing "
                          "shift_metrics.json, shortcut_probe.json, mc_dropout_summary.json, "
                          "abstention_metrics.json")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()
    assert len(args.seeds) == len(args.results_dirs)

    per_seed = []
    for seed, d in zip(args.seeds, args.results_dirs):
        d = Path(d)
        with open(d / "shift_metrics.json") as f:
            shift = json.load(f)
        with open(d / "shortcut_probe.json") as f:
            shortcut = json.load(f)
        with open(d / "mc_dropout_summary.json") as f:
            mcd = json.load(f)
        with open(d / "abstention_metrics.json") as f:
            abst = json.load(f)
        per_seed.append({
            "seed": seed,
            "shift_auroc": shift["auroc"], "shift_auprc": shift["auprc"], "shift_ece": shift["ece"],
            "shift_mean_pred_prob": shift["mean_predicted_prob"],
            "shortcut_probe_auroc": shortcut["probe_auroc"],
            "shortcut_probe_accuracy": shortcut["probe_accuracy"],
            "mc_dropout_correct_std": mcd["std_correct_vs_incorrect"]["correct_pred_mean_std"],
            "mc_dropout_incorrect_std": mcd["std_correct_vs_incorrect"]["incorrect_pred_mean_std"],
            "abstention_full_coverage_accuracy": abst["full_coverage_accuracy"],
            "abstention_auc_risk_uncertainty_ordered": abst["area_under_risk_coverage_uncertainty_ordered"],
            "abstention_auc_risk_random_ordered": abst["area_under_risk_coverage_random_ordered"],
        })

    keys = [k for k in per_seed[0] if k != "seed"]
    summary = {
        "n_seeds": len(args.seeds), "seeds": args.seeds, "per_seed": per_seed,
        **{k: mean_std([p[k] for p in per_seed]) for k in keys},
    }
    print(json.dumps(summary, indent=2))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "gate3_seed_ci.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nWrote {outdir / 'gate3_seed_ci.json'}")


if __name__ == "__main__":
    main()
