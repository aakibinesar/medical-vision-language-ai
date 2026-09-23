# Model Card — TrustMed-VLM

**Intended use.** Portfolio/research baseline comparing image-only chest X-ray
classification against a pretrained medical vision-language model (BiomedCLIP)
for image-report retrieval, with an explicit robustness/reliability evaluation
layer. **Not for clinical use.**

## Components

**Image-only baseline.** ResNet-18 or DenseNet-121 (ImageNet-pretrained),
transfer learning, single logit + BCEWithLogitsLoss. Trained on IU X-Ray
(primary) and separately evaluated on the Chest X-Ray Pneumonia set as a
cross-dataset distribution-shift check.

**Note on a mid-project pipeline fix.** All numbers below reflect a corrected
training pipeline: `dataset.py`'s training augmentation originally included
`RandomHorizontalFlip`, which is inappropriate for chest X-rays (anatomy
isn't left-right symmetric, and flipping mirrors embedded laterality
markers into nonsense) — plausibly why Grad-CAM was found attending to an
"L" marker in one case. Removing it and fully retraining every checkpoint
changed classification performance only within existing seed-to-seed noise,
*improved* calibration stability (seed-to-seed ECE std roughly halved, both
in- and out-of-distribution), and changed that specific Grad-CAM case from a
sharp asymmetric fixation on the "L" marker to broader symmetric attention
across both shoulder corners — a real change in behavior, not a clean fix
(the model still isn't attending to lung tissue there). A hard-coded
confusion-matrix labeling bug (always showed "Pneumonia" even for the
`Abnormal` label) and an unseeded, single-draw random-abstention baseline
(now averaged over 100 draws) were fixed at the same time. See
`kaggle/recheck-augmentation/`, `kaggle/recheck-gradcam-single/`, and
`kaggle/no-flip-full-ci/` for the full recheck.

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
  even improved (0.785 → 0.847), and ECE degraded more modestly than the
  original with-flip run (0.071 → 0.083, versus a near-tripling before) —
  discrimination survives distribution shift better than calibration does,
  though the gap is smaller and steadier now that training-time noise from
  the flip is gone.
- **Shortcut probe** (7,016 combined real images, 10x the small-scale n):
  a linear probe on frozen features separates IU X-Ray from Pneumonia
  images with AUROC = 0.99995, accuracy 99.6% (vs. 63.4% majority baseline)
  — confirms the small-scale perfect-separability finding was not a
  tiny-sample fluke, and is unaffected by the augmentation fix (as expected,
  since it probes representations, not augmentation-sensitive predictions).
- **MC-dropout uncertainty + abstention — reversed from small scale**: at
  full scale, predictive std is clearly higher for incorrect predictions
  (0.051) than correct ones (0.035), and uncertainty-ordered abstention
  performs well below random-order abstention (0.157 vs 0.264
  area-under-risk-coverage, the latter now averaged over 100 random
  permutations rather than one arbitrary draw). The small-scale null result
  was an artifact of too little training data, not a real property of the
  method.
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
- **Grad-CAM**: after removing `RandomHorizontalFlip` (see note above), the
  one overlay that previously showed a sharp, asymmetric hotspot centered on
  an "L" laterality marker now shows broader, symmetric attention across
  both shoulder corners instead — a real change in behavior, though not a
  clean fix: the model still isn't attending to lung tissue on this image,
  it just stopped uniquely fixating on one letter. A concrete, visible
  instance of the kind of shortcut the probe above found abstractly.

## Repeated-seed confidence intervals (n=3: seeds 42/43/44)

- **Classification (`results/metrics_seed_ci.json`)**: AUROC 0.772 ± 0.011
  and AUPRC 0.855 ± 0.008 are tight and essentially unchanged from the
  with-flip run (0.784 ± 0.010 / 0.864 ± 0.004) — within combined noise.
  **ECE after calibration is now 0.072 ± 0.022 — both a similar mean and a
  notably tighter spread than the with-flip run's 0.080 ± 0.045** (per-seed
  ECE: 0.071 / 0.095 / 0.051, vs. the old 0.040 / 0.071 / 0.130). Removing
  the flip didn't just fix a labeling/interpretability issue — it made
  calibration itself more seed-stable.
- **Contrastive projection fusion (`results/contrastive_projection.json`)**:
  unaffected by this fix (BiomedCLIP-based, not the CNN classifier) —
  confirmed robust — e.g. image→text Recall@5 7.41% ± 0.42% (trained) vs.
  3.64% (zero-shot, deterministic); the zero-shot baseline sits well
  outside the trained mean's range on every metric.
- **All four Gate 3 diagnostics (`results/gate3_seed_ci.json`) — every
  finding held up again after the fix, with calibration once more the
  most-improved metric**: shift AUROC 0.863±0.022 (still above each seed's
  own in-distribution AUROC, tighter than before); shift ECE 0.086±0.014
  (down from 0.117±0.026, and much tighter); shortcut probe AUROC
  0.99992±0.00003 (essentially seed-invariant, as expected - unaffected by
  the fix); MC-dropout std 0.036±0.005 (correct) vs. 0.049±0.004
  (incorrect), no overlap across seeds; abstention AUC-risk 0.166±0.008
  (uncertainty-ordered) vs. 0.261±0.005 (random-ordered, now itself an
  average over 100 permutations per seed rather than one arbitrary draw),
  again no overlap. Required training 5 new checkpoints (dropout-42,
  main/dropout-43, main/dropout-44) without the flip; the already-trained
  no-flip seed-42 main checkpoint was reused rather than retrained.
- The classification fusion baseline (already flagged as unreliable due to
  label leakage) and the full-scale error-analysis keyword check (already
  flagged as a blunt instrument) are still single-run against the seed-42
  checkpoint — not extended to repeated seeds, since a tighter variance
  estimate wouldn't change either conclusion.

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
- Backbone fine-tuning (unfreezing the last 2 transformer blocks of both
  towers, `contrastive_finetune.py`) was tried and didn't beat the frozen
  heads-only approach — a 40-epoch run shows a textbook overfitting curve
  (validation Recall@1 peaks at epoch 6, never recovers) and the best
  checkpoint only ties the frozen result at ~500x the compute cost. The
  experiment has its own confound though: it only ever trained against 128
  in-batch negatives per step versus the frozen approach's full 2,568-pair
  negative pool, so "backbone adaptation overfits" and "the training signal
  was too weak" can't be cleanly separated from this result alone. Tried to
  fix this directly with a larger batch (up to the full 2,568-example
  training set) - every attempt (2568, 512, 256) hit CUDA out-of-memory on
  the P100's 16GB; the real ceiling sits just above 128, too low to
  meaningfully close the gap. Documented as a hit hardware ceiling, not an
  abandoned thread - a MoCo-style memory bank of cached negatives is the
  well-scoped remaining future work. See `reports/technical_report.md`
  Section 7.
- The shortcut probe shows the representations encode acquisition source
  almost perfectly; this doesn't by itself prove the classifier's
  predictions depend on it, only that the information is present and usable.
- Repeated-seed CI now covers classification, contrastive fusion, and all
  four Gate 3 diagnostics; only the classification fusion baseline and
  error-analysis keyword check remain single-run (see above for why).
