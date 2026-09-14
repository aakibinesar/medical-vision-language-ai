# Model Card — TrustMed-VLM (Project 1)

**Intended use.** Portfolio/research baseline comparing image-only chest X-ray
classification against a pretrained medical vision-language model (BiomedCLIP)
for image-report retrieval, with an explicit robustness/reliability evaluation
layer. **Not for clinical use.**

## Components

**Image-only baseline.** ResNet-18 or DenseNet-121 (ImageNet-pretrained),
transfer learning, single logit + BCEWithLogitsLoss. Trained on IU X-Ray
(primary) and separately evaluated on the Chest X-Ray Pneumonia set as a
cross-dataset distribution-shift check.

**Multimodal baseline.** [BiomedCLIP](https://huggingface.co/microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224)
(PubMedBERT text tower + ViT-B/16 image tower, pretrained on PubMed Central
image-text pairs), used zero-shot for image<->report retrieval, and frozen
(no fine-tuning) as feature extractors for a logistic-regression fusion
classifier (`fusion_baseline.py`) — image-only vs. text-only vs. fusion
probes trained on identical splits, to directly test whether fusion adds
value beyond the best single modality.

## Metrics

- **Classification:** AUROC/AUPRC, confusion matrix, ROC curve, expected
  calibration error (ECE) before/after temperature scaling (`results/metrics.json`).
- **Retrieval:** Recall@1/5/10 for image->text and text->image
  (`results/retrieval_metrics.json`).
- **Text robustness:** Recall@K under text truncation, word-shuffling, removal,
  and mismatched pairing (`results/text_robustness.json`) — required per the
  project plan's "missing/noisy text" evaluation block. A healthy result shows
  recall collapsing toward chance under `empty`/`mismatched` corruption
  (confirms the model genuinely depends on report content) and degrading
  gracefully, not catastrophically, under `truncated`/`shuffled` (confirms
  some robustness to noisy real-world text).

**Interpretability.** Grad-CAM overlays on held-out test images
(`results/gradcam_examples/`), covering both correct and incorrect predictions.
IU X-Ray is CC BY-NC-ND 4.0 (NoDerivatives) — **do not publish Grad-CAM overlays
or other modified IU X-Ray images publicly** (public GitHub README, technical
report, slides) without the rightsholder's permission; use Pneumonia-dataset
examples (CC BY 4.0) for any public-facing qualitative figures instead.

## Label provenance and known risk

The `Abnormal` classification label on IU X-Ray is derived from the report's
`Problems`/MeSH field (0 if exactly "normal", 1 otherwise) — a heuristic, not
a clinician adjudication. Because the label comes from the same report whose
text is used for the multimodal comparison, there is a real risk of
**label-text leakage**: the VLM could appear to "understand" the image better
than it does simply because the report text it's matched against was the
source of the label itself. This is flagged explicitly per the project plan's
requirement to audit report-derived labels — it does not invalidate the
retrieval-based evaluation (which doesn't use the derived label at all) but
does mean the classification numbers should be read as an upper bound, not a
tight estimate, of image-only vs. multimodal advantage.

**This risk is now confirmed empirically, not just theoretical.** The fusion
baseline (below) shows text-only classification (AUROC 0.937) dramatically
outperforming image-only (0.653) — consistent with the text classifier
partly reading its own label back out of correlated report text, not
demonstrating that reports carry far more diagnostic signal than images.

## Trustworthy-evaluation results (real data, small-scale)

All results below are from a small, real (non-synthetic) 240/80/80-study
local IU X-Ray subsample — see `reports/technical_report.md` Section 4 for
the full-scale-run caveat. Full numbers: `results/shift_metrics.json`,
`results/shortcut_probe.json`, `results/mc_dropout_summary.json`,
`results/abstention_metrics.json`.

- **Distribution shift** (evaluated on the real Pneumonia test set, n=624):
  AUROC held up (0.749 → 0.771) but ECE roughly tripled (0.094 → 0.263) —
  discrimination survived the shift, calibration didn't.
- **Shortcut probe**: a linear probe on frozen features separates IU X-Ray
  from Pneumonia images with AUROC = 1.0 — the representations strongly
  encode acquisition source (hospital/scanner), a real shortcut-learning
  flag that tempers how the shift result above should be read.
- **MC-dropout uncertainty + abstention**: predictive std was nearly
  identical for correct vs. incorrect predictions, and uncertainty-ordered
  abstention performed no better than random. Honest null result at this
  scale — needs re-checking on the full-scale run before concluding
  anything about whether MC-dropout uncertainty is useful here.
- **Fusion baseline** (image-only 0.653 vs. text-only 0.937 vs. fusion 0.940
  AUROC, same real split as the CNN baseline): fusion technically edges out
  the best unimodal score, but by 0.003 AUROC on 80 test examples — not a
  real effect. The dominant result is text-only crushing image-only, which
  is the label-leakage risk above showing up empirically rather than
  evidence of genuine multimodal benefit. See
  `reports/technical_report.md` Section 7 for the full discussion.

## Limitations

- All classification/shift/shortcut/uncertainty numbers to date are from
  the small local CPU subsample described above, not the full 2,568-study
  training set — the full-scale Kaggle GPU run is still pending and should
  anchor final reported numbers.
- IU X-Ray split is by study `uid`, which is effectively patient-level for
  this corpus, but this hasn't been independently verified against a true
  patient identifier.
- Chest X-Ray Pneumonia (used only as the cross-dataset shift test) has no
  patient IDs at all — its own split is image-wise, a pre-existing limitation
  disclosed in `DATASET_DATASHEET.md`.
- The fusion baseline's classification comparison is confounded by
  text-label leakage (see above) — it does not cleanly answer "does fusion
  help," only "does fusion beat a leaky text baseline" (barely, within
  noise). A clean fusion test needs either a non-text-derived label or
  evaluation on the retrieval task instead.
