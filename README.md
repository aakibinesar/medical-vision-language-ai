# TrustMed-VLM: Trustworthy Medical Vision-Language AI

Project 1 of a two-project PhD-application research programme (Project 2,
PathoSpatial-GNN, is a separate spatial-biology/graph-learning repo, currently
in data-feasibility scouting only).

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
medical-vision-language-ai-portfolio/
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
│   └── aggregate_seed_metrics.py    # mean/std across repeated-seed training runs
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
report. Two kernels — `kaggle/full-run/` (training + Gate 3 + error
analysis) and `kaggle/fusion-retrieval/` (the genuine fusion test). See
`kaggle/README.md` for the full walkthrough and gotchas; short version:

```bash
# 1. Push src/ as a private Kaggle dataset (shared by both kernels)
cd src && kaggle datasets create -p .   # (needs a dataset-metadata.json; see kaggle/README.md)

# 2. Push and run either pipeline as a GPU kernel
cd ../kaggle/full-run && kaggle kernels push -p .            # training + Gate 3 + error analysis
cd ../fusion-retrieval && kaggle kernels push -p .            # genuine fusion test

# 3. Poll status, then fetch output once complete
kaggle kernels status <owner>/trustmed-vlm-full-run
kaggle kernels output <owner>/trustmed-vlm-full-run -p out/   # or see kaggle/README.md for the
                                                                 # Windows encoding-bug workaround
```

## Status against the Project 1 stage gates

All gates below reflect the **full-scale** run (2,568/549/549-study real IU
X-Ray split, DenseNet-121, 224px, Kaggle GPU) unless noted. An earlier
small-scale pass (240/80/80 studies, local CPU) validated the pipeline
first; see `reports/technical_report.md` for the full small-scale-vs-
full-scale comparison — several results changed meaningfully at scale, most
notably MC-dropout uncertainty (null → clearly informative) and the fusion
verdict (nominal small-scale "win" → full-scale loss, confirming that "win"
was noise).

- [x] **Gate 0** (data feasibility): both datasets loaded from real
      downloaded data — IU X-Ray (3,666 studies split by `uid`) and
      Pneumonia. See `DATASET_DATASHEET.md`.
- [x] **Gate 1** (unimodal baseline): full-scale DenseNet-121 — test AUROC
      0.792, AUPRC 0.865, ECE 0.052→0.040 after calibration (seed 42). All
      improved over the small-scale pipeline-check numbers (0.749/0.810/0.094).
      **Repeated-seed check (n=3):** AUROC 0.784±0.010 and AUPRC 0.864±0.004
      are tight and reliable; **ECE after calibration is actually 0.080±0.045
      — the reported 0.040 was the best of three seeds, not typical**, and
      the best-AUROC seed had the worst calibration. See
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
      at full scale. AUROC survives the pneumonia shift (0.792→0.833) but
      calibration degrades (ECE 0.040→0.131); a linear probe separates the
      two datasets' learned features almost perfectly (AUROC 0.9997 on
      7,016 combined real images — confirms the small-scale perfect score
      wasn't a fluke); MC-dropout uncertainty **reverses** the small-scale
      null result — at full scale it's clearly informative (uncertainty-
      ordered abstention roughly halves risk vs. random). See
      `reports/technical_report.md` Section 5 and `MODEL_CARD.md`.
- [x] **Error analysis**: full 549-study test set, with a systematic
      keyword check (not just eyeballing the most-confident cases) across
      every FP/FN — which showed the small-scale qualitative narrative
      ("FN=subtle chronic findings, FP=postsurgical hardware") was
      real-but-overstated: both patterns are still visible but explain a
      minority of cases at full scale. Reported as a methodological lesson,
      not quietly dropped — see `reports/technical_report.md` Section 8.
- [x] **Gate 4** (supervisor-ready): technical report and model card hold
      real full-scale results throughout, the genuine fusion test, and
      repeated-seed confidence intervals (n=3) for both headline results —
      which caught a real issue (calibration is far less seed-stable than
      discrimination; the reported ECE was a best-case draw, now corrected
      to 0.080±0.045). Remaining: a final read-through polish, and
      optionally extending seed repeats to the Gate 3 diagnostics, which
      are still single-run.
