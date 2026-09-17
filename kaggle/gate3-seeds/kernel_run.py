"""Extend repeated-seed confidence intervals to the Gate 3 diagnostics
(shift, shortcut, MC-dropout/abstention), which were single-run against
the seed-42 checkpoint only. Reuses the 3 already-trained main checkpoints
(seeds 42/43/44, uploaded as trustmed-vlm-checkpoints) for shift_eval and
shortcut_probe - no retraining needed there. MC-dropout needs a
dropout-enabled checkpoint per seed; only seed 42's exists already, so this
trains 2 more (seeds 43, 44, --dropout-p 0.3) here.
"""
import json
import os
import subprocess
import sys
import time

print("=== environment setup ===", flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "numpy==1.26.4"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                 "torch==2.3.1", "torchvision==0.18.1",
                 "--index-url", "https://download.pytorch.org/whl/cu121"], check=True)

import torch
print("torch", torch.__version__, "cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))

SRC = "/kaggle/input/datasets/aakibinesar/trustmed-vlm-src"
CKPT = "/kaggle/input/datasets/aakibinesar/trustmed-vlm-checkpoints"
WORK = "/kaggle/working"
IU_ROOT = "/kaggle/input/datasets/raddar/chest-xrays-indiana-university"
IU_IMG_ROOT = f"{IU_ROOT}/images/images_normalized"
PNEU_DATA_ROOT = "/kaggle/input/datasets/paultimothymooney/chest-xray-pneumonia/chest_xray"
PNEU_IMG_ROOT = "/kaggle/input/datasets/paultimothymooney/chest-xray-pneumonia"

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
run("prepare_pneumonia_csv", [
    "prepare_csv.py", "--data-root", PNEU_DATA_ROOT, "--out-dir", "data_pneumonia",
])

# --- Train the 2 missing dropout-enabled checkpoints ---
for seed in (43, 44):
    run(f"train_dropout_seed{seed}", [
        "train.py",
        "--train-csv", "data_iuxray/train.csv", "--val-csv", "data_iuxray/val.csv",
        "--img-root", IU_IMG_ROOT, "--label-columns", "Abnormal",
        "--epochs", "15", "--batch-size", "32", "--img-size", "224",
        "--model", "densenet121", "--dropout-p", "0.3", "--seed", str(seed),
        "--outdir", f"outputs_dropout_seed{seed}",
    ])

MAIN_CKPTS = {42: f"{CKPT}/seed42_main.pt", 43: f"{CKPT}/seed43_main.pt", 44: f"{CKPT}/seed44_main.pt"}
DROPOUT_CKPTS = {42: f"{CKPT}/seed42_dropout.pt",
                  43: "outputs_dropout_seed43/best.pt", 44: "outputs_dropout_seed44/best.pt"}

# --- Shift + shortcut for all 3 seeds (reusing existing main checkpoints) ---
for seed, ckpt in MAIN_CKPTS.items():
    run(f"shift_eval_seed{seed}", [
        "shift_eval.py", "--checkpoint", ckpt,
        "--shift-csv", "data_pneumonia/test.csv", "--shift-img-root", PNEU_IMG_ROOT,
        "--shift-target-column", "Pneumonia", "--outdir", f"results_seed{seed}",
    ])
    run(f"shortcut_probe_seed{seed}", [
        "shortcut_probe.py", "--checkpoint", ckpt,
        "--csv-a", "data_iuxray/train.csv", "--img-root-a", IU_IMG_ROOT,
        "--label-column-a", "Abnormal", "--name-a", "iu_xray",
        "--csv-b", "data_pneumonia/train.csv", "--img-root-b", PNEU_IMG_ROOT,
        "--label-column-b", "Pneumonia", "--name-b", "pneumonia",
        "--outdir", f"results_seed{seed}",
    ])

# --- MC-dropout + abstention for all 3 seeds ---
for seed, ckpt in DROPOUT_CKPTS.items():
    run(f"mc_dropout_seed{seed}", [
        "uncertainty_mc_dropout.py", "--csv", "data_iuxray/test.csv", "--img-root", IU_IMG_ROOT,
        "--checkpoint", ckpt, "--n-samples", "20", "--outdir", f"results_seed{seed}",
    ])
    run(f"abstention_seed{seed}", [
        "abstention_eval.py", "--mc-dropout-csv", f"results_seed{seed}/mc_dropout_predictions.csv",
        "--outdir", f"results_seed{seed}",
    ])

with open(os.path.join(WORK, "pipeline_status.json"), "w") as f:
    json.dump(STATUS, f, indent=2)
print("\n=== PIPELINE STATUS ===")
print(json.dumps(STATUS, indent=2))
n_failed = sum(1 for v in STATUS.values() if not v["ok"])
print(f"\n{len(STATUS) - n_failed}/{len(STATUS)} steps succeeded.")
