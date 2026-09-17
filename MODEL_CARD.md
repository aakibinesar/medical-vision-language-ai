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
image-text pairs), used three ways: zero-shot for image<->report retrieval;
frozen as feature extractors for a logistic-regression fusion classifier
(`fusion_baseline.py`, confounded by label leakage — see below); and frozen
with small trained linear projection heads on top, via a contrastive loss
on the real image-report pairs (`contrastive_projection.py`) — the
project's one genuinely unconfounded fusion test, since it never touches
the derived classification label.

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

**This risk is confirmed empirically at full scale, and gets worse with
more data, not better.** The fusion baseline (below) shows text-only
classification (AUROC 0.957 at full scale) dramatically outperforming
image-only (0.752) — an even larger gap than the small-scale pass (0.937 vs
0.653). If this were genuine diagnostic signal in the prose rather than
leakage, the gap would be expected to narrow, not widen, with 10x more
training data — this pattern is much more consistent with leakage than
with reports simply being more informative than images.

## Trustworthy-evaluation results (real data, full scale)

Full-scale run: DenseNet-121, 224px, full real IU X-Ray split (2,568 train /
549 val / 549 test), executed via the Kaggle API. An earlier small-scale
pass (240/80/80 studies, local CPU) validated the pipeline first; see
`reports/technical_report.md` for the full small-scale-vs-full-scale
comparison — several results changed meaningfully at scale, most notably
MC-dropout uncertainty (null result → clearly informative) and the fusion
verdict (nominal small-scale "win" → full-scale loss). Full numbers:
`results/shift_metrics.json`, `results/shortcut_probe.json`,
`results/mc_dropout_summary.json`, `results/abstention_metrics.json`.

- **Distribution shift** (real Pneumonia test set, n=624): AUROC held up,
  even improved (0.792 → 0.833), but ECE nearly tripled (0.040 → 0.131) —
  discrimination survives distribution shift better than calibration does,
  replicating the small-scale finding at a larger, more reliable margin.
- **Shortcut probe** (7,016 combined real images, 10x the small-scale n):
  a linear probe on frozen features separates IU X-Ray from Pneumonia
  images with AUROC = 0.9997, accuracy 99.8% (vs. 63.4% majority baseline)
  — confirms the small-scale perfect-separability finding was not a
  tiny-sample fluke.
- **MC-dropout uncertainty + abstention — reversed from small scale**: at
  full scale, predictive std is clearly higher for incorrect predictions
  (0.047) than correct ones (0.029), and uncertainty-ordered abstention
  roughly halves risk vs. random-order abstention (0.170 vs 0.323
  area-under-risk-coverage). The small-scale null result was an artifact of
  too little training data, not a real property of the method.
- **Fusion baseline (classification, confounded)** (image-only 0.752 vs.
  text-only 0.957 vs. fusion 0.955 AUROC): fusion does **not** beat
  text-only at full scale — this reverses the small-scale pass's nominal
  (and, in retrospect, noise-level) "win." The Gate 2 bar is not met on
  this classification setup. See `reports/technical_report.md` Section 7.
- **Contrastive projection (retrieval, unconfounded) — the real fusion
  win**: training small linear projection heads with a contrastive loss on
  the real image-report training pairs (never touching the derived label)
  roughly **doubles Recall@5 and Recall@10** over zero-shot BiomedCLIP in
  both directions (e.g. image→text Recall@5 3.64%→7.83%), with real gains
  at Recall@1 too. Validated by a healthy training curve (best checkpoint
  at epoch 28/200, well before the loss-minimizing late epochs) rather than
  a small-sample fluke — this is the project's cleanest piece of positive
  multimodal evidence. `results/contrastive_projection.json`.
- **Grad-CAM**: one overlay's hottest region centers on an "L" laterality
  marker rather than lung tissue — a concrete, visible instance of the kind
  of shortcut the probe above found abstractly. Not a systematic finding by
  itself (one image), but a good, precise follow-up target.

## Repeated-seed confidence intervals (n=3: seeds 42/43/44)

- **Classification (`results/metrics_seed_ci.json`)**: AUROC 0.784 ± 0.010
  and AUPRC 0.864 ± 0.004 are tight and reliable. **ECE after calibration
  is 0.080 ± 0.045 — the originally-reported 0.040 was the best of three
  seeds, not typical**, and the seed with the best AUROC (0.788) had the
  worst calibration (0.130) of the three. Report 0.080 ± 0.045 as the
  honest calibration number going forward, not 0.040.
- **Contrastive projection fusion (`results/contrastive_projection.json`)**:
  confirmed robust — e.g. image→text Recall@5 7.41% ± 0.42% (trained) vs.
  3.64% (zero-shot, deterministic); the zero-shot baseline sits well
  outside the trained mean's range on every metric.
- Every other evaluation here (shift, shortcut probe, MC-dropout/
  abstention, error analysis, classification fusion) is still a single run
  against the seed-42 checkpoint, not extended to repeated seeds.

## Limitations

- IU X-Ray split is by study `uid`, which is effectively patient-level for
  this corpus, but this hasn't been independently verified against a true
  patient identifier.
- Chest X-Ray Pneumonia (used only as the cross-dataset shift test) has no
  patient IDs at all — its own split is image-wise, a pre-existing limitation
  disclosed in `DATASET_DATASHEET.md`.
- The classification fusion baseline is confounded by text-label leakage
  (see above) — it does not cleanly answer "does fusion help," only "does
  fusion beat a leaky text baseline" (no, at full scale). The retrieval-
  based contrastive projection test above is the clean answer instead, and
  it's positive.
- The contrastive projection only trains small linear heads on top of a
  fully frozen backbone — it doesn't establish how much further gains
  might come from fine-tuning more of the network (a small MLP head, or
  unfreezing late backbone layers), just that even this minimal amount of
  training on real pairs already helps.
- The shortcut probe shows the representations encode acquisition source
  almost perfectly; this doesn't by itself prove the classifier's
  predictions depend on it, only that the information is present and usable.
- Repeated-seed CI exists for the two headline results (n=3 each) but not
  for the diagnostic Gate 3 evaluations, which remain single-run.
