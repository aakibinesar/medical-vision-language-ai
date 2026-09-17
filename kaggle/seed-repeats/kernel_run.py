"""Repeated-seed training for a real confidence interval on the Gate 1
headline classification numbers, instead of the single-seed point estimate
from the main full-run kernel (which used seed 42, already in
results/metrics.json). Trains seeds 43 and 44 here (2 more), so combined
with the existing seed-42 result there's n=3 for a mean/std estimate -
deliberately not re-running seed 42, since that compute is already spent
and its result already saved.

Same hyperparameters as the original full-scale run: DenseNet-121, 224px,
15 epochs, batch 32, on the full real IU X-Ray split.
"""
import json
import os
import subprocess
import sys
import time

print("=== environment setup ===", flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                 "numpy==1.26.4"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                 "torch==2.3.1", "torchvision==0.18.1",
                 "--index-url", "https://download.pytorch.org/whl/cu121"], check=True)

import torch
print("torch", torch.__version__, "cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))

SRC = "/kaggle/input/datasets/aakibinesar/trustmed-vlm-src"
WORK = "/kaggle/working"
IU_ROOT = "/kaggle/input/datasets/raddar/chest-xrays-indiana-university"
IU_IMG_ROOT = f"{IU_ROOT}/images/images_normalized"

os.chdir(WORK)
STATUS = {}


def run(step_name, args):
    print(f"\n=== {step_name} ===", flush=True)
    print("+ python", " ".join(args), flush=True)
    t0 = time.time()
    try:
        subprocess.run([sys.executable, os.path.join(SRC, args[0])] + args[1:],
                        check=True, cwd=WORK)
        STATUS[step_name] = {"ok": True, "seconds": round(time.time() - t0, 1)}
    except subprocess.CalledProcessError as e:
        print(f"!!! {step_name} FAILED: {e}", flush=True)
        STATUS[step_name] = {"ok": False, "seconds": round(time.time() - t0, 1), "error": str(e)}


run("prepare_iuxray_csv", [
    "prepare_iuxray_csv.py",
    "--reports-csv", f"{IU_ROOT}/indiana_reports.csv",
    "--projections-csv", f"{IU_ROOT}/indiana_projections.csv",
    "--out-dir", "data_iuxray",
])

SEEDS = [43, 44]
for seed in SEEDS:
    run(f"train_seed{seed}", [
        "train.py",
        "--train-csv", "data_iuxray/train.csv", "--val-csv", "data_iuxray/val.csv",
        "--img-root", IU_IMG_ROOT, "--label-columns", "Abnormal",
        "--epochs", "15", "--batch-size", "32", "--img-size", "224",
        "--model", "densenet121", "--seed", str(seed), "--outdir", f"outputs_seed{seed}",
    ])
    run(f"eval_seed{seed}", [
        "eval.py", "--val-csv", "data_iuxray/val.csv", "--test-csv", "data_iuxray/test.csv",
        "--img-root", IU_IMG_ROOT, "--checkpoint", f"outputs_seed{seed}/best.pt",
        "--outdir", f"results_seed{seed}",
    ])

with open(os.path.join(WORK, "pipeline_status.json"), "w") as f:
    json.dump(STATUS, f, indent=2)
print("\n=== PIPELINE STATUS ===")
print(json.dumps(STATUS, indent=2))
