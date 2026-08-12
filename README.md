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
│   └── text_robustness.py           # missing/noisy/mismatched-text robustness test
├── notebooks/kaggle_baseline_training.ipynb   # run this on Kaggle for real GPU training
├── data/                       # not committed; see data/README.md
├── results/                    # metrics, plots, Grad-CAM examples (generated)
├── reports/technical_report.md
├── MODEL_CARD.md
└── DATASET_DATASHEET.md
```

## Why local + Kaggle

Developed on a laptop with a 2GB-VRAM GPU (NVIDIA MX450) — enough to write and
smoke-test code, not enough to train at real resolution or run BiomedCLIP over
a full dataset at speed. Local work stays on tiny synthetic data; real runs
happen on Kaggle's free GPU.

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

## Quickstart — real training (Kaggle, recommended)

1. Upload this repo to GitHub (or upload `src/` as a Kaggle Dataset/utility script).
2. Create a Kaggle Notebook, add the "Chest X-Ray Pneumonia" dataset (image-only
   baseline) and/or "Chest X-rays (Indiana University)" dataset (multimodal).
3. `notebooks/kaggle_baseline_training.ipynb` covers the image-only baseline end
   to end. The IU X-Ray multimodal notebook is the next thing to build once the
   real dataset has been inspected (see Status below).
4. For the multimodal scripts: `python src/prepare_iuxray_csv.py ...`, then
   `python src/retrieval_baseline.py ...` and `python src/text_robustness.py ...`
   against the real data.
5. Download `results/` from the notebook output and commit them to this repo.

## Status against the Project 1 stage gates

- [x] Gate 0 (data feasibility): pneumonia loader done; IU X-Ray loader written,
      not yet run against real downloaded data.
- [x] Gate 1 (unimodal baseline): ResNet-18/DenseNet-121 trainer, calibration,
      Grad-CAM — implemented and smoke-tested.
- [ ] Gate 2 (multimodal value): zero-shot BiomedCLIP retrieval baseline and
      text-robustness test implemented and smoke-tested; not yet run on real
      IU X-Ray data, and no fine-tuned fusion baseline yet.
- [ ] Gate 3 (trustworthy evaluation): calibration/ECE done; distribution-shift
      (pneumonia cross-dataset test), shortcut-learning probes, MC-dropout/ensemble
      uncertainty, and abstention are not yet implemented.
- [ ] Gate 4 (supervisor-ready): technical report, model card, and results are
      still placeholders pending a real training run.
