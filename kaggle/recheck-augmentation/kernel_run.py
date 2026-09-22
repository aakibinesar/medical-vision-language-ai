"""Re-checks two bugfixes from the pipeline audit against real full-scale
data, both needing the actual IU X-Ray images (not available locally):

1. eval.py's confusion-matrix plot had hard-coded "Pneumonia" axis labels
   even when evaluating IU X-Ray's `Abnormal` label - fixed to use the real
   label name. This kernel regenerates the plot correctly.
2. dataset.py's training-time RandomHorizontalFlip is removed (chest X-ray
   anatomy isn't left-right symmetric, and it mirrors laterality markers
   into nonsense) - plausibly why Grad-CAM was previously found attending
   to an "L" marker instead of lung tissue. Retrains just the seed-42 main
   checkpoint (same hyperparameters as the original full-run kernel) to see
   whether that Grad-CAM behavior changes, before deciding whether the fix
   is worth propagating through a full seed-repeats/gate3-seeds re-run.

Single seed only, deliberately - this is a targeted recheck, not a new CI.
"""
import json
import os
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
                 "open_clip_torch", "transformers"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "numpy==1.26.4"], check=True)
subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                 "torch==2.3.1", "torchvision==0.18.1",
                 "--index-url", "https://download.pytorch.org/whl/cu121"], check=True)

import torch
print("torch", torch.__version__, "cuda available:", torch.cuda.is_available())

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

run("train_main_no_flip", [
    "train.py",
    "--train-csv", "data_iuxray/train.csv", "--val-csv", "data_iuxray/val.csv",
    "--img-root", IU_IMG_ROOT, "--label-columns", "Abnormal",
    "--epochs", "15", "--batch-size", "32", "--img-size", "224",
    "--model", "densenet121", "--outdir", "outputs_full_no_flip",
])

run("eval_main_no_flip", [
    "eval.py", "--val-csv", "data_iuxray/val.csv", "--test-csv", "data_iuxray/test.csv",
    "--img-root", IU_IMG_ROOT, "--checkpoint", "outputs_full_no_flip/best.pt",
    "--outdir", "results_no_flip",
])
run("gradcam_main_no_flip", [
    "gradcam.py", "--csv", "data_iuxray/test.csv", "--img-root", IU_IMG_ROOT,
    "--checkpoint", "outputs_full_no_flip/best.pt", "--n-examples", "6",
    "--out", "results_no_flip/gradcam_examples",
])

with open(os.path.join(WORK, "pipeline_status.json"), "w") as f:
    json.dump(STATUS, f, indent=2)
print("\n=== PIPELINE STATUS ===")
print(json.dumps(STATUS, indent=2))
