"""Follow-up to the frozen-backbone genuine fusion test in
../fusion-retrieval/kernel_run.py: fine-tunes the last 2 transformer blocks
of BOTH BiomedCLIP towers (image ViT + text BERT) instead of keeping the
whole backbone frozen, using contrastive_finetune.py. Answers the open
question in contrastive_projection.py's own limitations section - does
letting the backbone adapt help beyond training just the two linear
projection heads, or does it overfit on ~2,568 real pairs? Reuses the same
tested src/ scripts as every other kernel.

A first pilot (seed 42, 12 epochs, batch 64) was inconclusive (loss still
dropping, no plateau). A second attempt (seed 42, 40 epochs, batch 128)
gave a clear textbook overfitting curve - validation Recall@1 peaked at
epoch 6/40 and never recovered - and the best checkpoint only tied the
frozen-heads-only result. But that run has its own confound: at batch 128
it only ever trained against 127 in-batch negatives per step, versus the
frozen approach's full 2,568-pair negative pool every step - a much weaker
training signal, entirely independent of whether backbone adaptation
itself is a good idea. Three attempts to fix this all hit the same wall -
CUDA OOM on the P100's 16GB, with 2 unfrozen ViT + 2 unfrozen BERT blocks:
--batch-size 2568 (full training set, matching the frozen approach's
negative pool exactly - "Tried to allocate 5.79 GiB"), --batch-size 512
("Tried to allocate 1.50 GiB"), and --batch-size 256 ("Tried to allocate
768 MiB"). The shrinking overflow suggests the true ceiling sits just above
128 - maybe 150-190 - but even hitting it exactly would only take the
negative pool from 127 to ~180, nowhere near enough to meaningfully close
the gap to the frozen approach's 2,568. Diminishing returns: decided not to
keep guessing narrower batch sizes for a fix that wouldn't resolve the
confound even in the best case. **Documented as a hit hardware ceiling
instead** (see `reports/technical_report.md` Section 7) - closing this
confound properly would need a different technique entirely (e.g. a
MoCo-style memory bank of cached past-batch negatives, which decouples
negative-pool size from GPU memory), not a bigger batch. This script is
left running its last successful, already-reported configuration
(seed 42, 40 epochs, batch 128) below, for reproducibility.
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
