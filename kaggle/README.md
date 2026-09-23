# Full-scale runs via the Kaggle API

This is what actually produced the full-scale results in `results/` and
`reports/technical_report.md` — not a browser notebook. Kept here for
reproducibility, and because the local `src/*.py` scripts each kernel runs
are the same ones used everywhere else in this repo (no separate/duplicated
notebook code to drift out of sync, which is what happened to an earlier
pneumonia-only notebook this repo used to ship — retired once the primary
dataset moved to IU X-Ray).

All kernels below share the same `src/*.py` dataset (`trustmed-vlm-src`) —
update it once (`kaggle datasets version`), re-run whichever kernel(s) you
need.

## `full-run/` — training, calibration, Gate 3, error analysis

Data prep, full-scale training (main + MC-dropout checkpoints),
calibration/eval, Grad-CAM, the classification fusion baseline, and the
whole Gate 3 + error-analysis suite — each step wrapped so one failure
doesn't abort the rest. Needs `trustmed-vlm-src` + IU X-Ray + Pneumonia as
dataset sources.

## `fusion-retrieval/` — the genuine, unconfounded fusion test

Data prep + `contrastive_projection.py` only (see that script's docstring
and `reports/technical_report.md` Section 7 for what it does and why it's
the clean fusion evidence the classification test in `full-run/` isn't).
Lighter than `full-run/` — no CNN training, just BiomedCLIP embedding
extraction plus a couple hundred fast epochs on tiny linear heads. Needs
`trustmed-vlm-src` + IU X-Ray only (no Pneumonia — this test never uses it).

`contrastive_projection.py` itself supports multiple seeds in one
invocation (`--seeds 42 43 44`) since embeddings are deterministic and only
need extracting once — `fusion-retrieval/` uses this directly rather than
needing a separate kernel per seed.

## `fusion-finetune/` — does letting the backbone adapt beat frozen heads?

Follow-up to `fusion-retrieval/`: fine-tunes the last 2 transformer blocks
of BOTH BiomedCLIP towers (`contrastive_finetune.py`) instead of keeping
the backbone fully frozen, with a much smaller backbone learning rate than
the projection heads. Result: no - the training curve is a textbook
overfitting signature (validation Recall@1 peaks at epoch 6 of 40, never
recovers), and even the best early-stopped checkpoint only ties the frozen
approach's performance at roughly 500x the compute cost. See
`reports/technical_report.md`. Needs `trustmed-vlm-src` + IU X-Ray only.

## `no-flip-full-ci/` — full repeated-seed retrain after a pipeline audit fix

A full pipeline audit found `dataset.py`'s training-time
`RandomHorizontalFlip` was inappropriate for chest X-rays (mirrors
laterality markers into nonsense) - plausibly the cause of a Grad-CAM
overlay fixating on an "L" marker instead of lung tissue, alongside two
other real bugs (see the gotchas below). This kernel is the full
propagation: reuses the already-trained no-flip seed-42 main checkpoint,
trains the 5 remaining ones without the flip (dropout-42, main/dropout-43,
main/dropout-44), and reruns every Gate 1/Gate 3 step for all 3 seeds. Does
*not* touch anything BiomedCLIP-based (retrieval, fusion, contrastive
projection/fine-tune) - those don't use the CNN classifier or its
augmentation, so they're unaffected.

It was reached in two cheaper steps first (both since retired, since this
kernel superseded them): a single-seed pilot retrained just the seed-42
checkpoint to confirm the fix was safe (classification metrics unaffected)
before committing to a full re-run, and a second, inference-only step
re-rendered Grad-CAM for the *exact* "L marker" image under the no-flip
checkpoint using `gradcam.py`'s `--include-path` flag (added for this
purpose - forces a specific image into the sample rather than relying on
the random sample to happen to include it again).

Result: classification metrics unchanged within seed noise; calibration got
*more* seed-stable (ECE std roughly halved, both in- and
out-of-distribution); the specific Grad-CAM case changed from sharp
asymmetric marker-fixation to broader symmetric shoulder-corner attention -
a real change, not a clean fix. See `MODEL_CARD.md` and
`reports/technical_report.md` Section 6.

## How it works (any kernel above)

1. `src/*.py` is pushed as a private Kaggle dataset (`kaggle datasets create`,
   or `kaggle datasets version` to update an existing one) so the kernel
   runs the actual tested scripts via subprocess, not a copy.
2. `kernel_run.py` is pushed as a GPU kernel (`kaggle kernels push -p .` from
   inside that kernel's own folder) with that folder's `kernel-metadata.json`
   declaring its dataset sources.
3. Status/output is polled and fetched via the API (`kaggle kernels status` /
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
  characters (pip's own progress-bar output, for instance). Workaround:
  `python kaggle/fetch_kernel_log.py <owner/kernel-slug> <out-path>` calls
  the same underlying API directly and writes the log with
  `encoding="utf-8"` itself, rather than using `kernels_output`/
  `kaggle kernels output` as-is.
- Two real bugs in this repo's own code only surfaced at Kaggle/GPU scale
  and are now fixed in `src/`: `gradcam.py`'s DenseNet-121 hook (PyTorch
  autograd "view + inplace" conflict from a module-level full backward hook
  colliding with DenseNet's in-place ReLU) and `abstention_eval.py` (relied
  on `np.trapezoid`, which doesn't exist before numpy 2.0 — Kaggle's numpy
  was older; fixed with a manual trapezoidal calculation).
- A later full pipeline audit (not Kaggle-specific, but only checkable
  against real data) found three more: `eval.py`'s confusion-matrix plot
  had hard-coded "Pneumonia" axis labels even when evaluating a different
  label; `dataset.py` trained with `RandomHorizontalFlip`, inappropriate
  for chest X-rays; `abstention_eval.py`'s random-order baseline used one
  hard-coded, unseeded permutation instead of averaging over many. All
  three fixed in `src/`; `gradcam.py` also gained an `--include-path` flag
  to force a specific image into the sample, for exact before/after
  comparisons (used to re-render the "L marker" case under the fix — see
  `no-flip-full-ci/` above).
