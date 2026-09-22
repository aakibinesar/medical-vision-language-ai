"""Inference-only recheck: re-renders Grad-CAM for one specific image -
3089_IM-1444-1001.dcm.png, the exact case the model card describes as
attending to an "L" laterality marker instead of lung tissue - using the
seed-42 checkpoint trained WITHOUT RandomHorizontalFlip (uploaded as
trustmed-vlm-no-flip-ckpt), for a direct before/after comparison against
the original (with-flip) overlay already in results/gradcam_examples/.
No training here - just data prep and one gradcam.py call.
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
CKPT = "/kaggle/input/datasets/aakibinesar/trustmed-vlm-no-flip-ckpt/no_flip_seed42_main.pt"

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

run("gradcam_single_recheck", [
    "gradcam.py", "--csv", "data_iuxray/test.csv", "--img-root", IU_IMG_ROOT,
    "--checkpoint", CKPT, "--n-examples", "1",
    "--include-path", "3089_IM-1444-1001.dcm.png",
    "--out", "results_single/gradcam_examples",
])

with open(os.path.join(WORK, "pipeline_status.json"), "w") as f:
    json.dump(STATUS, f, indent=2)
print("\n=== PIPELINE STATUS ===")
print(json.dumps(STATUS, indent=2))
