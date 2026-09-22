"""Uncertainty-aware abstention / risk-coverage analysis — the exact
"uncertainty-aware abstention" evaluation named in the project plan's
novelty target. Consumes uncertainty_mc_dropout.py's output: sorts examples
by predictive uncertainty (std across MC-dropout passes) and asks "if the
model were allowed to abstain (say 'refer to a human') on its most uncertain
X% of cases, how much would accuracy improve on what's left?"

A healthy result: risk (error rate) on the answered subset drops as coverage
shrinks — abstaining on uncertain cases should concentrate errors into the
abstained-on set. If risk barely changes with coverage, the uncertainty
estimate isn't actually tracking correctness, which is itself an important
(negative) finding to report honestly.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def risk_coverage_curve(y_true, y_pred, uncertainty, coverages):
    """uncertainty: higher = less confident. Abstain on the most uncertain
    (1 - coverage) fraction; report accuracy/error on what's left."""
    order = np.argsort(uncertainty)  # ascending uncertainty = most confident first
    n = len(y_true)
    rows = []
    for cov in coverages:
        k = max(1, int(round(cov * n)))
        kept = order[:k]
        acc = float((y_pred[kept] == y_true[kept]).mean())
        rows.append({"coverage": cov, "n_answered": k, "accuracy": acc, "risk": 1 - acc})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mc-dropout-csv", required=True,
                     help="Output of uncertainty_mc_dropout.py (path,y_true,mean_prob,std_prob)")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=42, help="Base seed for the random-order baseline")
    ap.add_argument("--n-random-draws", type=int, default=100,
                     help="Random-order baseline is averaged over this many independent "
                          "permutations, not a single fixed draw - one arbitrary permutation "
                          "can be a lucky or unlucky draw on its own, which previously made "
                          "the 'random' baseline noisier than the seed-to-seed std alone let on.")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    df = pd.read_csv(args.mc_dropout_csv)
    y_true = df["y_true"].values
    y_pred = (df["mean_prob"].values >= args.threshold).astype(int)

    coverages = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]

    curve_uncertainty = risk_coverage_curve(y_true, y_pred, df["std_prob"].values, coverages)

    def auc_risk(curve):
        # Manual trapezoidal rule - avoids relying on np.trapz (removed in
        # numpy 2.0) vs np.trapezoid (added in numpy 2.0, absent before),
        # which differ depending on which numpy is installed (e.g. this
        # repo's local venv vs. Kaggle's preinstalled version).
        covs = [r["coverage"] for r in curve][::-1]
        risks = [r["risk"] for r in curve][::-1]
        area = 0.0
        for i in range(1, len(covs)):
            area += (risks[i] + risks[i - 1]) / 2 * (covs[i] - covs[i - 1])
        return float(area)

    # Baseline comparison: abstaining by random order instead of by uncertainty.
    # If the uncertainty-based curve isn't meaningfully better than random
    # abstention, the uncertainty estimate isn't earning its keep. Averaged
    # over many independent permutations rather than one fixed draw - a
    # single arbitrary permutation can itself be a lucky or unlucky draw,
    # which previously left the "random" baseline carrying its own unreported
    # source of noise on top of genuine model-to-model (seed) variance.
    rng = np.random.default_rng(args.seed)
    random_draw_aucs = []
    random_draw_curves = []
    for _ in range(args.n_random_draws):
        random_order_score = rng.permutation(len(y_true)).astype(float)
        curve = risk_coverage_curve(y_true, y_pred, random_order_score, coverages)
        random_draw_curves.append(curve)
        random_draw_aucs.append(auc_risk(curve))

    # Representative curve for the plot/JSON: risk at each coverage level
    # averaged across all draws (the AUC summary below uses the per-draw
    # AUCs directly, which is the statistically cleaner quantity).
    curve_random = []
    for i, cov in enumerate(coverages):
        accs = [c[i]["accuracy"] for c in random_draw_curves]
        curve_random.append({
            "coverage": cov, "n_answered": random_draw_curves[0][i]["n_answered"],
            "accuracy": float(np.mean(accs)), "risk": float(1 - np.mean(accs)),
        })

    summary = {
        "full_coverage_accuracy": float((y_pred == y_true).mean()),
        "area_under_risk_coverage_uncertainty_ordered": auc_risk(curve_uncertainty),
        "area_under_risk_coverage_random_ordered": float(np.mean(random_draw_aucs)),
        "area_under_risk_coverage_random_ordered_std": float(np.std(random_draw_aucs)),
        "n_random_draws": args.n_random_draws,
        "curve_uncertainty_ordered": curve_uncertainty,
        "curve_random_ordered": curve_random,
    }
    lower_is_better_note = ("Lower area-under-risk-coverage is better. The uncertainty-ordered "
                             "curve should score lower than the random-ordered one if the MC-dropout "
                             "uncertainty is actually informative about correctness.")
    summary["interpretation"] = lower_is_better_note
    print(json.dumps(summary, indent=2))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "abstention_metrics.json", "w") as f:
        json.dump(summary, f, indent=2)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot([r["coverage"] for r in curve_uncertainty], [r["risk"] for r in curve_uncertainty],
            marker="o", label="abstain by uncertainty")
    ax.plot([r["coverage"] for r in curve_random], [r["risk"] for r in curve_random],
            marker="o", linestyle="--", label="abstain randomly (baseline)")
    ax.set_xlabel("Coverage (fraction of cases answered)")
    ax.set_ylabel("Risk (error rate on answered cases)")
    ax.set_title("Risk-coverage curve")
    ax.legend()
    ax.invert_xaxis()
    fig.tight_layout()
    fig.savefig(outdir / "risk_coverage_curve.png", dpi=150)
    print(f"Wrote abstention_metrics.json and risk_coverage_curve.png to {outdir}/")


if __name__ == "__main__":
    main()
