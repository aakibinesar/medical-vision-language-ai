"""Full propagation of the RandomHorizontalFlip removal (dataset.py) through
every checkpoint and Gate 1/Gate 3 result that depends on the trained CNN
classifier - replacing the with-flip results committed so far, not just a
single-seed recheck (see kaggle/README.md for how this kernel came about;
the cheaper pilot kernels that led here have since been retired).

Does NOT touch anything BiomedCLIP-based (retrieval, text-robustness,
classification fusion baseline, contrastive projection/fine-tune) - none of
those use the CNN classifier or its augmentation, so they're unaffected and
not rerun here.

Reuses the seed-42 main checkpoint already trained without the flip
(uploaded as trustmed-vlm-no-flip-ckpt) - no need to pay for that training
again. Trains the 5 remaining checkpoints this project's full Gate 1 +
Gate 3 CI needs: dropout-42, main-43, main-44, dropout-43, dropout-44.
Same hyperparameters as every prior run: DenseNet-121, 224px, 15 epochs,
batch 32.
"""
import json
import os
import shutil
import subprocess
import sys
import time

print("=== diagnostic /kaggle/input listing ===", flush=True)
for root, dirs, files in os.walk("/kaggle/input"):
    depth = root.count(os.sep) - "/kaggle/input".count(os.sep)
    if depth > 2:
        dirs[:] = []
        continue
    print(root, "->", dirs, files[:8], flush=True)

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
NO_FLIP_CKPT = "/kaggle/input/datasets/aakibinesar/trustmed-vlm-no-flip-ckpt/no_flip_seed42_main.pt"
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

# Reuse the already-trained seed-42 main (no-flip) checkpoint - no need to pay
# for that training again.
os.makedirs("outputs_seed42", exist_ok=True)
shutil.copy(NO_FLIP_CKPT, "outputs_seed42/best.pt")
STATUS["reuse_seed42_main_checkpoint"] = {"ok": True, "seconds": 0.0}

# --- Train the 5 remaining no-flip checkpoints ---
run("train_dropout_seed42", [
    "train.py",
    "--train-csv", "data_iuxray/train.csv", "--val-csv", "data_iuxray/val.csv",
    "--img-root", IU_IMG_ROOT, "--label-columns", "Abnormal",
    "--epochs", "15", "--batch-size", "32", "--img-size", "224",
    "--model", "densenet121", "--dropout-p", "0.3", "--seed", "42",
    "--outdir", "outputs_dropout_seed42",
])
for seed in (43, 44):
    run(f"train_main_seed{seed}", [
        "train.py",
        "--train-csv", "data_iuxray/train.csv", "--val-csv", "data_iuxray/val.csv",
        "--img-root", IU_IMG_ROOT, "--label-columns", "Abnormal",
        "--epochs", "15", "--batch-size", "32", "--img-size", "224",
        "--model", "densenet121", "--seed", str(seed), "--outdir", f"outputs_seed{seed}",
    ])
    run(f"train_dropout_seed{seed}", [
        "train.py",
        "--train-csv", "data_iuxray/train.csv", "--val-csv", "data_iuxray/val.csv",
        "--img-root", IU_IMG_ROOT, "--label-columns", "Abnormal",
        "--epochs", "15", "--batch-size", "32", "--img-size", "224",
        "--model", "densenet121", "--dropout-p", "0.3", "--seed", str(seed),
        "--outdir", f"outputs_dropout_seed{seed}",
    ])

MAIN_CKPTS = {42: "outputs_seed42/best.pt", 43: "outputs_seed43/best.pt", 44: "outputs_seed44/best.pt"}
DROPOUT_CKPTS = {42: "outputs_dropout_seed42/best.pt", 43: "outputs_dropout_seed43/best.pt",
                  44: "outputs_dropout_seed44/best.pt"}

# --- Gate 1: eval (now with the fixed confusion-matrix labels) for all 3 seeds ---
for seed, ckpt in MAIN_CKPTS.items():
    run(f"eval_seed{seed}", [
        "eval.py", "--val-csv", "data_iuxray/val.csv", "--test-csv", "data_iuxray/test.csv",
        "--img-root", IU_IMG_ROOT, "--checkpoint", ckpt, "--outdir", f"results_seed{seed}",
    ])

# Grad-CAM for seed 42 only (matches original design) - same 6-example sample
# plus the specific "L marker" case, so it lands in the same output set this
# time instead of needing a separate targeted rerun.
run("gradcam_seed42", [
    "gradcam.py", "--csv", "data_iuxray/test.csv", "--img-root", IU_IMG_ROOT,
    "--checkpoint", MAIN_CKPTS[42], "--n-examples", "6",
    "--include-path", "3089_IM-1444-1001.dcm.png",
    "--out", "results_seed42/gradcam_examples",
])

# --- Gate 3: shift + shortcut (main checkpoints), MC-dropout + abstention (dropout checkpoints) ---
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

for seed, ckpt in DROPOUT_CKPTS.items():
    run(f"mc_dropout_seed{seed}", [
        "uncertainty_mc_dropout.py", "--csv", "data_iuxray/test.csv", "--img-root", IU_IMG_ROOT,
        "--checkpoint", ckpt, "--n-samples", "20", "--outdir", f"results_seed{seed}",
    ])
    run(f"abstention_seed{seed}", [
        "abstention_eval.py", "--mc-dropout-csv", f"results_seed{seed}/mc_dropout_predictions.csv",
        "--outdir", f"results_seed{seed}",
    ])

# --- Error analysis (seed 42 main only, matches original single-run design) ---
run("error_analysis", [
    "error_analysis.py", "--csv", "data_iuxray/test.csv", "--img-root", IU_IMG_ROOT,
    "--checkpoint", MAIN_CKPTS[42], "--outdir", "results_seed42",
])

with open(os.path.join(WORK, "pipeline_status.json"), "w") as f:
    json.dump(STATUS, f, indent=2)
print("\n=== PIPELINE STATUS ===")
print(json.dumps(STATUS, indent=2))
n_failed = sum(1 for v in STATUS.values() if not v.get("ok"))
print(f"\n{len(STATUS) - n_failed}/{len(STATUS)} steps succeeded.")
