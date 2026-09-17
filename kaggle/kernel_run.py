"""Full-scale TrustMed-VLM run: data prep, training (main + MC-dropout),
eval/calibration, Grad-CAM, multimodal retrieval/fusion, and the full Gate 3
trustworthy-evaluation suite (shift, shortcut, uncertainty, abstention) plus
error analysis. Every step is the actual tested repo code (from the
trustmed-vlm-src dataset), not a re-implementation.

Each step is wrapped so a failure doesn't abort the whole run - later steps
that don't depend on the failed one still get a chance to produce results.
"""
import json
import os
import subprocess
import sys
import time

print("=== Step -1: diagnostic /kaggle/input listing ===", flush=True)
for root, dirs, files in os.walk("/kaggle/input"):
    depth = root.count(os.sep) - "/kaggle/input".count(os.sep)
    if depth > 2:
        dirs[:] = []
        continue
    print(root, "->", dirs, files[:8], flush=True)

print("=== Step 0: environment setup ===", flush=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                 "open_clip_torch", "transformers"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                 "numpy==1.26.4"], check=True)
# Kaggle's default preinstalled torch doesn't support this account's assigned
# P100 GPU (sm_60 dropped from recent cu128 builds) - pin a compatible build.
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                 "torch==2.3.1", "torchvision==0.18.1",
                 "--index-url", "https://download.pytorch.org/whl/cu121"], check=True)

import torch
print("torch", torch.__version__, "cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device:", torch.cuda.get_device_name(0))
    _ = (torch.randn(100, 100, device="cuda") @ torch.randn(100, 100, device="cuda")).sum()
    torch.cuda.synchronize()
    print("GPU matmul sanity check OK")

SRC = "/kaggle/input/datasets/aakibinesar/trustmed-vlm-src"
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


# --- Data prep ---
run("prepare_iuxray_csv", [
    "prepare_iuxray_csv.py",
    "--reports-csv", f"{IU_ROOT}/indiana_reports.csv",
    "--projections-csv", f"{IU_ROOT}/indiana_projections.csv",
    "--out-dir", "data_iuxray",
])
run("prepare_pneumonia_csv", [
    "prepare_csv.py", "--data-root", PNEU_DATA_ROOT, "--out-dir", "data_pneumonia",
])

# --- Gate 1: full-scale training (main + MC-dropout variant) ---
run("train_main", [
    "train.py",
    "--train-csv", "data_iuxray/train.csv", "--val-csv", "data_iuxray/val.csv",
    "--img-root", IU_IMG_ROOT, "--label-columns", "Abnormal",
    "--epochs", "15", "--batch-size", "32", "--img-size", "224",
    "--model", "densenet121", "--outdir", "outputs_full",
])
run("train_dropout", [
    "train.py",
    "--train-csv", "data_iuxray/train.csv", "--val-csv", "data_iuxray/val.csv",
    "--img-root", IU_IMG_ROOT, "--label-columns", "Abnormal",
    "--epochs", "15", "--batch-size", "32", "--img-size", "224",
    "--model", "densenet121", "--dropout-p", "0.3", "--outdir", "outputs_full_dropout",
])

run("eval_main", [
    "eval.py", "--val-csv", "data_iuxray/val.csv", "--test-csv", "data_iuxray/test.csv",
    "--img-root", IU_IMG_ROOT, "--checkpoint", "outputs_full/best.pt", "--outdir", "results_full",
])
run("gradcam_main", [
    "gradcam.py", "--csv", "data_iuxray/test.csv", "--img-root", IU_IMG_ROOT,
    "--checkpoint", "outputs_full/best.pt", "--n-examples", "6",
    "--out", "results_full/gradcam_examples",
])

# --- Gate 2: multimodal (full test set, no subsampling needed on GPU) ---
run("retrieval_baseline", [
    "retrieval_baseline.py", "--csv", "data_iuxray/test.csv", "--img-root", IU_IMG_ROOT,
    "--outdir", "results_full",
])
run("text_robustness", [
    "text_robustness.py", "--csv", "data_iuxray/test.csv", "--img-root", IU_IMG_ROOT,
    "--outdir", "results_full",
])
run("fusion_baseline", [
    "fusion_baseline.py",
    "--train-csv", "data_iuxray/train.csv", "--val-csv", "data_iuxray/val.csv",
    "--test-csv", "data_iuxray/test.csv", "--img-root", IU_IMG_ROOT,
    "--label-column", "Abnormal", "--outdir", "results_full",
])

# --- Gate 3: trustworthy evaluation ---
run("shift_eval", [
    "shift_eval.py", "--checkpoint", "outputs_full/best.pt",
    "--shift-csv", "data_pneumonia/test.csv", "--shift-img-root", PNEU_IMG_ROOT,
    "--shift-target-column", "Pneumonia",
    "--in-distribution-metrics", "results_full/metrics.json", "--outdir", "results_full",
])
run("shortcut_probe", [
    "shortcut_probe.py", "--checkpoint", "outputs_full/best.pt",
    "--csv-a", "data_iuxray/train.csv", "--img-root-a", IU_IMG_ROOT,
    "--label-column-a", "Abnormal", "--name-a", "iu_xray",
    "--csv-b", "data_pneumonia/train.csv", "--img-root-b", PNEU_IMG_ROOT,
    "--label-column-b", "Pneumonia", "--name-b", "pneumonia",
    "--outdir", "results_full",
])
run("mc_dropout_uncertainty", [
    "uncertainty_mc_dropout.py", "--csv", "data_iuxray/test.csv", "--img-root", IU_IMG_ROOT,
    "--checkpoint", "outputs_full_dropout/best.pt", "--n-samples", "20", "--outdir", "results_full",
])
run("abstention_eval", [
    "abstention_eval.py", "--mc-dropout-csv", "results_full/mc_dropout_predictions.csv",
    "--outdir", "results_full",
])

# --- Error analysis ---
run("error_analysis", [
    "error_analysis.py", "--csv", "data_iuxray/test.csv", "--img-root", IU_IMG_ROOT,
    "--checkpoint", "outputs_full/best.pt", "--outdir", "results_full",
])

with open(os.path.join(WORK, "pipeline_status.json"), "w") as f:
    json.dump(STATUS, f, indent=2)
print("\n=== PIPELINE STATUS ===")
print(json.dumps(STATUS, indent=2))
n_failed = sum(1 for v in STATUS.values() if not v["ok"])
print(f"\n{len(STATUS) - n_failed}/{len(STATUS)} steps succeeded.")
