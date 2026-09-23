# TrustMed-VLM: Trustworthy Medical Vision-Language AI

**Research question:** Can a pretrained medical vision-language model provide
more reliable and transferable predictions than conventional image-only
models, particularly under limited labels and distribution shift?

**Not for clinical use.** Educational/portfolio project only.

## Task and datasets

Binary "abnormal vs. normal" chest X-ray classification and image-report
retrieval, with an explicit image-only vs. image+text comparison:

| Dataset | Role | Paired text? | Patient-level split? |
|---|---|---|---|
| [IU X-Ray / Open-i](https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university) | Primary | Yes | Yes (by study `uid`) |
| [Chest X-Ray Pneumonia](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia) | Cross-dataset shift test | No | No (disclosed limitation) |
| MIMIC-CXR | Planned upgrade, pending PhysioNet credentialing | Yes | Yes |

See `data/README.md` for how to get each one, including the MIMIC-CXR
credentialing steps (the one part of this plan that can't be automated —
requires your own PhysioNet identity verification).

## Repository structure

```
medical-vision-language-ai/
├── src/
│   ├── dataset.py                  # CXRDataset + transforms (classification)
│   ├── clip_utils.py                # shared BiomedCLIP embedding extraction
│   ├── prepare_csv.py               # pneumonia dataset -> train/val/test.csv
│   ├── prepare_iuxray_csv.py        # IU X-Ray -> train/val/test.csv (path,report_text,Abnormal)
│   ├── make_synthetic_smoke_data.py            # tiny fake image-only set for smoke tests
│   ├── make_synthetic_multimodal_smoke_data.py # tiny fake image+text set for smoke tests
│   ├── train.py                     # ResNet-18 / DenseNet-121 baseline trainer
│   ├── eval.py                      # AUROC/AUPRC, confusion matrix, ROC, calibration/ECE
│   ├── gradcam.py                   # Grad-CAM overlays on test images
│   ├── retrieval_baseline.py        # zero-shot BiomedCLIP image<->report retrieval
│   ├── fusion_baseline.py           # image-only vs text-only vs fusion classification probes (confounded)
│   ├── contrastive_projection.py    # genuine fusion test: contrastive retrieval fine-tuning (unconfounded)
│   ├── text_robustness.py           # missing/noisy/mismatched-text robustness test
│   ├── shift_eval.py                # cross-dataset distribution-shift test
│   ├── shortcut_probe.py            # dataset-of-origin linear probe (shortcut-learning check)
│   ├── uncertainty_mc_dropout.py, abstention_eval.py  # MC-dropout uncertainty + risk-coverage
│   ├── error_analysis.py            # per-example predictions + FP/FN report-text inspection
│   ├── aggregate_seed_metrics.py    # mean/std across repeated-seed training runs (Gate 1)
│   └── aggregate_gate3_seed_metrics.py  # mean/std across repeated-seed Gate 3 diagnostics
├── kaggle/                     # the actual scripts that ran the full-scale GPU job (see kaggle/README.md)
├── data/                       # not committed; see data/README.md
├── results/                    # metrics, plots, Grad-CAM examples (generated)
├── reports/technical_report.md
├── MODEL_CARD.md
└── DATASET_DATASHEET.md
```

## Why local + Kaggle

Developed on a laptop with a 2GB-VRAM GPU (NVIDIA MX450) — enough to write and
smoke-test code (and even train small real-data checkpoints slowly on CPU),
not enough for full-scale training or fast BiomedCLIP inference over a full
dataset. Local work stays on tiny synthetic data or small real subsamples for
pipeline validation; the full-scale, headline-number run happens on Kaggle's
free GPU — driven end-to-end via the Kaggle API (`kaggle/`), not a browser
notebook. See `kaggle/README.md` for how, including the real environment
gotchas hit doing it (GPU/PyTorch version mismatch, dataset-mount path
differences, a Windows-only SDK bug).

## Quickstart — local smoke test (pipeline correctness only, no downloads)

```bash
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install -r requirements.txt
cd src

# Image-only classification pipeline
python make_synthetic_smoke_data.py --out-dir ../data_smoke --n-per-class 20 --img-size 64
python train.py --train-csv ../data_smoke/train.csv --val-csv ../data_smoke/val.csv \
    --img-root ../data_smoke --epochs 2 --batch-size 8 --img-size 64 --model resnet18 \
    --outdir ../outputs_smoke
python eval.py --val-csv ../data_smoke/val.csv --test-csv ../data_smoke/test.csv \
    --img-root ../data_smoke --checkpoint ../outputs_smoke/best.pt --outdir ../results_smoke
python gradcam.py --csv ../data_smoke/test.csv --img-root ../data_smoke \
    --checkpoint ../outputs_smoke/best.pt --out ../results_smoke/gradcam_examples

# Multimodal retrieval + text-robustness pipeline (downloads BiomedCLIP weights, ~380MB, once)
python make_synthetic_multimodal_smoke_data.py --out-dir ../data_multimodal_smoke --n-per-class 8
python retrieval_baseline.py --csv ../data_multimodal_smoke/train.csv --img-root ../data_multimodal_smoke
python text_robustness.py --csv ../data_multimodal_smoke/train.csv --img-root ../data_multimodal_smoke
```

Both pipelines have been run and verified working end-to-end, including a
real BiomedCLIP load and inference (not mocked).

## Quickstart — real full-scale training (Kaggle API)

This is what actually produced everything in `results/` and the technical
report. Four kernels — `kaggle/full-run/` (training + Gate 1-3 + error
analysis, plus retrieval/text-robustness/classification-fusion), `kaggle/
fusion-retrieval/` (the genuine, unconfounded fusion test, multi-seed
capable), `kaggle/fusion-finetune/` (the backbone fine-tuning follow-up —
a negative result), and `kaggle/no-flip-full-ci/` (full repeated-seed
Gate 1/Gate 3 retrain after a pipeline audit fix). See `kaggle/README.md`
for the full walkthrough, gotchas, and why earlier kernels were retired.
Short version:

```bash
# 1. Push src/ as a private Kaggle dataset (shared by every kernel)
cd src && kaggle datasets create -p .   # (needs a dataset-metadata.json; see kaggle/README.md)

# 2. Push and run whichever pipeline(s) you need as a GPU kernel
cd ../kaggle/full-run && kaggle kernels push -p .           # training + Gate 1-3 + error analysis
cd ../fusion-retrieval && kaggle kernels push -p .          # genuine fusion test
cd ../fusion-finetune && kaggle kernels push -p .           # backbone fine-tuning follow-up
cd ../no-flip-full-ci && kaggle kernels push -p .           # full retrain after the pipeline audit fix

# 3. Poll status, then fetch output once complete
kaggle kernels status <owner>/trustmed-vlm-full-run
kaggle kernels output <owner>/trustmed-vlm-full-run -p out/   # or see kaggle/README.md for the
                                                                 # Windows encoding-bug workaround
```

## Status against the project's stage gates

All gates below reflect the **full-scale** run (2,568/549/549-study real IU
X-Ray split, DenseNet-121, 224px, Kaggle GPU) unless noted. An earlier
small-scale pass (240/80/80 studies, local CPU) validated the pipeline
first; see `reports/technical_report.md` for the full small-scale-vs-
full-scale comparison — several results changed meaningfully at scale, most
notably MC-dropout uncertainty (null → clearly informative) and the fusion
verdict (nominal small-scale "win" → full-scale loss, confirming that "win"
was noise).

**Mid-project pipeline audit and fix:** a full pass over every script found
and fixed three real issues — a training-time `RandomHorizontalFlip`
inappropriate for chest X-rays (plausibly the cause of a Grad-CAM overlay
fixating on an "L" laterality marker), a hard-coded confusion-matrix label
bug, and an unseeded single-draw random-abstention baseline. All checkpoints
and Gate 1/Gate 3 results below were fully retrained/rerun after the fix;
see `MODEL_CARD.md`'s pipeline-fix note and `kaggle/no-flip-full-ci/` for
details.

- [x] **Gate 0** (data feasibility): both datasets loaded from real
      downloaded data — IU X-Ray (3,666 studies split by `uid`) and
      Pneumonia. See `DATASET_DATASHEET.md`.
- [x] **Gate 1** (unimodal baseline): full-scale DenseNet-121 — test AUROC
      0.785, AUPRC 0.859, ECE 0.069→0.071 after calibration (seed 42, no-flip
      pipeline). Essentially unchanged from the original with-flip run
      (0.792/0.865/0.040) within seed-to-seed noise.
      **Repeated-seed check (n=3):** AUROC 0.772±0.011 and AUPRC 0.855±0.008
      remain tight; **ECE after calibration is 0.072±0.022 — a similar mean
      to the with-flip run (0.080±0.045) but with roughly half the
      seed-to-seed spread** (0.071/0.095/0.051 per seed vs. the old
      0.040/0.071/0.130) — removing the flip made calibration more
      seed-stable, not just fixed an interpretability issue. See
      `results/metrics_seed_ci.json` and `reports/technical_report.md`
      Section 4.
- [x] **Gate 2** (multimodal value) — **met, via the unconfounded test**:
      zero-shot BiomedCLIP retrieval (full 549-study test set, several
      times chance level), text-robustness (collapses toward chance under
      empty/mismatched text), a classification fusion probe comparison
      (confounded by label leakage — fusion does *not* beat text-only,
      0.955 vs 0.957 AUROC; the small-scale "win" there was noise), and a
      genuine, non-label-confounded fusion test: contrastive projection
      heads trained on the real image-report pairs (never touching the
      derived label) **roughly double Recall@5/@10** over zero-shot
      BiomedCLIP in both directions, with real Recall@1 gains too — a
      healthy training curve (best checkpoint at epoch 28/200) rules out
      overfitting, and a **repeated-seed check (n=3) confirms it's real**:
      std is small (0.4-1.1pp) relative to the ~2x effect, and zero-shot
      sits outside the trained mean's range on every metric. This is the
      project's cleanest positive multimodal result and the one that
      actually satisfies the Gate 2 bar. See
      `reports/technical_report.md` Section 7.
- [x] **Gate 3** (trustworthy evaluation): calibration/ECE, cross-dataset
      shift, shortcut probe, and MC-dropout uncertainty/abstention all run
      at full scale on the no-flip pipeline. AUROC survives the pneumonia
      shift (0.785→0.847) and calibration degrades more modestly than
      before (ECE 0.071→0.083, versus a near-tripling pre-fix); a linear
      probe separates the two datasets' learned features almost perfectly
      (AUROC 0.99995 on 7,016 combined real images, unaffected by the fix as
      expected); MC-dropout uncertainty is clearly informative (uncertainty-
      ordered abstention risk 0.157 vs. 0.264 random-ordered, the latter now
      averaged over 100 permutations rather than one arbitrary draw).
      **Repeated-seed check (n=3, `results/gate3_seed_ci.json`): every
      finding held up again after the fix, and calibration got tighter**:
      shift ECE 0.086±0.014 (down from 0.117±0.026 pre-fix); shortcut probe
      AUROC essentially seed-invariant (0.99992±0.00003); MC-dropout's
      correct-vs-incorrect std gap and abstention's uncertainty-vs-random
      gap both hold with no overlap across all three seeds. See
      `reports/technical_report.md` Section 5 and `MODEL_CARD.md`.
- [x] **Error analysis**: full 549-study test set, with a systematic
      keyword check (not just eyeballing the most-confident cases) across
      every FP/FN — which showed the small-scale qualitative narrative
      ("FN=subtle chronic findings, FP=postsurgical hardware") was
      real-but-overstated: both patterns are still visible but explain a
      minority of cases at full scale. Reported as a methodological lesson,
      not quietly dropped — see `reports/technical_report.md` Section 8.
- [x] **Gate 4** (write-up complete) — **complete**: technical report and
      model card hold real full-scale results throughout, the genuine
      fusion test, and repeated-seed confidence intervals (n=3) for every
      headline number and all four Gate 3 diagnostics. The classification
      CI caught a real issue (calibration is far less seed-stable than
      discrimination; the reported single-seed ECE was a best-case draw);
      the Gate 3 CI, by contrast, confirmed every finding held up with
      nothing to correct. Final polish pass done: wrote the previously-
      unfilled Motivation section, added a "Key findings at a glance"
      summary to the technical report, fixed stale cross-references, added
      `.gitattributes` and `LICENSE` (MIT, code only — not the datasets),
      removed an unused dependency. A later full pipeline audit found and
      fixed three real issues (inappropriate training augmentation, a
      confusion-matrix labeling bug, an unseeded abstention baseline) and
      every checkpoint/result above was fully retrained/rerun afterward —
      see the pipeline-fix note above and in `MODEL_CARD.md`.
