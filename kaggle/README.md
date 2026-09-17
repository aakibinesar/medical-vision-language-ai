# Full-scale run via the Kaggle API

This is what actually produced the full-scale results in `results/` and
`reports/technical_report.md` — not a browser notebook. Kept here for
reproducibility, and because the local `src/*.py` scripts it runs are the
same ones used everywhere else in this repo (no separate/duplicated notebook
code to drift out of sync, which is what happened to an earlier
pneumonia-only notebook this repo used to ship — retired once the primary
dataset moved to IU X-Ray).

## How it works

1. `src/*.py` is pushed as a private Kaggle dataset (`kaggle datasets create`)
   so the kernel runs the actual tested scripts via subprocess, not a copy.
2. `kernel_run.py` is pushed as a GPU kernel (`kaggle kernels push`) with
   `kernel-metadata.json` declaring three dataset sources: the src dataset,
   IU X-Ray, and Pneumonia.
3. The kernel runs the full pipeline end to end: data prep, full-scale
   training (main + MC-dropout checkpoints), calibration/eval, Grad-CAM,
   multimodal retrieval/fusion, and the whole Gate 3 + error-analysis suite
   — each step wrapped so one failure doesn't abort the rest.
4. Status/output is polled and fetched via the API (`kaggle kernels status` /
   `kernels output`) once complete.

## Real gotchas hit doing this (so you don't have to rediscover them)

- **Kaggle's default preinstalled PyTorch may not support your assigned
  GPU.** This account got a Tesla P100 (compute capability sm_60), but the
  preinstalled torch build only supported sm_70+. Fix: pin an older,
  broadly-compatible build first thing in the kernel —
  `pip install torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121`.
  The `--accelerator` flag on `kaggle kernels push` did not change which
  GPU got assigned for this account.
- **GPU kernels mount datasets at a different path than CPU kernels did**
  in this account's testing: `/kaggle/input/datasets/<owner>/<slug>/`, not
  `/kaggle/input/<slug>/`. Verify with a directory-listing diagnostic before
  assuming either.
- **A freshly-created private dataset isn't immediately usable.** Pushing a
  kernel that references a dataset created moments earlier can fail with
  "No such file or directory" for every file in it — Kaggle needs a short
  processing delay after `datasets create` before the dataset is reliably
  mountable. Wait a bit, or verify with a diagnostic kernel first.
- **The Kaggle Python SDK has a Windows-only bug** in `kernels_output`: it
  opens the downloaded log file with the OS default encoding (cp1252 on
  Windows) instead of UTF-8, which crashes if the log contains non-Latin1
  characters (pip's own progress-bar output, for instance). Workaround: call
  the same underlying API directly and write the log with
  `encoding="utf-8"` yourself, rather than using `kernels_output`/
  `kaggle kernels output` as-is.
- Two real bugs in this repo's own code only surfaced at Kaggle/GPU scale
  and are now fixed in `src/`: `gradcam.py`'s DenseNet-121 hook (PyTorch
  autograd "view + inplace" conflict from a module-level full backward hook
  colliding with DenseNet's in-place ReLU) and `abstention_eval.py` (relied
  on `np.trapezoid`, which doesn't exist before numpy 2.0 — Kaggle's numpy
  was older; fixed with a manual trapezoidal calculation).
