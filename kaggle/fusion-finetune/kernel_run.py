"""Follow-up to the frozen-backbone genuine fusion test in
../fusion-retrieval/kernel_run.py: fine-tunes the last 2 transformer blocks
of BOTH BiomedCLIP towers (image ViT + text BERT) instead of keeping the
whole backbone frozen, using contrastive_finetune.py. Answers the open
question in contrastive_projection.py's own limitations section - does
letting the backbone adapt help beyond training just the two linear
projection heads, or does it overfit on ~2,568 real pairs? Reuses the same
tested src/ scripts as every other kernel.

A first pilot (seed 42, 12 epochs, batch 64, ~77 min) found training loss
still dropping with no plateau and validation Recall@1 not yet trending
upward - inconclusive, not converged. This run extends it: more epochs
(40, up from 12) so it actually has room to converge, and a bigger batch
(128, up from 64) to strengthen the in-batch negative pool used by the
contrastive loss each step (a batch of 64 negatives is much weaker than the
frozen approach's full 2,568-pair negative pool - the most likely reason
the pilot underperformed on image->text specifically). Still seed 42 only,
for direct comparison against the pilot - a 3-seed CI is a separate,
later decision once this shows whether fine-tuning is worth it at all.
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

run("contrastive_finetune", [
    "contrastive_finetune.py",
    "--train-csv", "data_iuxray/train.csv", "--val-csv", "data_iuxray/val.csv",
    "--test-csv", "data_iuxray/test.csv", "--img-root", IU_IMG_ROOT,
    "--unfreeze-layers", "2", "--epochs", "40",
    "--batch-size", "128", "--eval-batch-size", "32",
    "--seeds", "42",
    "--outdir", "results_full",
])

with open(os.path.join(WORK, "pipeline_status.json"), "w") as f:
    json.dump(STATUS, f, indent=2)
print("\n=== PIPELINE STATUS ===")
print(json.dumps(STATUS, indent=2))
