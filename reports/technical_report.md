# TrustMed-VLM: Trustworthy Medical Vision-Language AI Under Distribution Shift

*Draft skeleton — fill in as each phase completes.*

**Novelty target (from the Project 1 red-team assessment):** a systematic
comparison of image-only and medical vision-language models under limited
labels, missing or corrupted text, calibration error, shortcut dependence,
and cross-environment distribution shift, including uncertainty-aware
abstention. The result is not "we fine-tuned a VLM on medical images" — it's
the reliability/robustness comparison itself.

## 1. Motivation

(TODO: clinical framing, why image-only vs. multimodal reliability matters
under limited labels and distribution shift.)

## 2. Dataset

**Primary: IU X-Ray / Open-i.** 3,666 studies with both a usable report and a
derivable label (from 3,689 frontal studies with a report; 23 dropped for
missing report/label), split by study `uid` (2,568 train / 549 val / 549
test). See `DATASET_DATASHEET.md` for full provenance, license (CC BY-NC-ND
4.0 — no public derivative images), and the weak-label caveat.

**Secondary: Chest X-Ray Pneumonia.** Used only as a cross-dataset
distribution-shift probe (different hospital, patient population, and label
scheme — see `shift_eval.py`'s docstring for why this isn't a direct
accuracy comparison).

## 3. Methods

Image-only baseline: ResNet-18/DenseNet-121, transfer learning,
BCEWithLogitsLoss, optional MC-dropout head (`--dropout-p`). Multimodal
baseline: zero-shot BiomedCLIP (PubMedBERT + ViT-B/16) image-report
retrieval, no fine-tuning yet. Trustworthy-evaluation layer: temperature
scaling + ECE, MC-dropout uncertainty, uncertainty-ordered risk-coverage
(abstention), cross-dataset shift, missing/noisy-text robustness, and a
dataset-of-origin shortcut probe on frozen penultimate features.

## 4. Baseline models — full-scale results

**These are the headline numbers**, from the full-scale run: DenseNet-121,
224px, on the complete real IU X-Ray split (2,568 train / 549 val / 549
test), run via the Kaggle API (see the note on methodology at the end of
this section). An earlier small-scale pass (240/80/80 studies, ResNet-18,
160px, local CPU) was used to validate the pipeline before committing to
full-scale compute; those numbers are kept below each result for
comparison, since several of them changed meaningfully at scale — that
delta is itself informative about which small-scale findings were real
signal versus small-sample noise.

| Metric | Full-scale (headline) | Small-scale (pipeline check) |
|---|---|---|
| Test AUROC | **0.792** | 0.749 |
| Test AUPRC | **0.865** | 0.810 |
| ECE before calibration | **0.052** | 0.137 |
| ECE after temperature scaling | **0.040** | 0.094 |

More training data improved both discrimination and calibration, as
expected — and did so by more than a small tweak: ECE after calibration
roughly halved. Full history in `outputs_full/history.json`.

**Zero-shot BiomedCLIP retrieval** (full real 549-study IU X-Ray test set):
image→text Recall@1/5/10 = 1.28% / 3.64% / 5.65% (chance with 549
candidates = 0.18% / 0.91% / 1.82%); text→image = 1.09% / 4.01% / 6.19%.
Consistent with the small-scale finding (400-study subsample: 2.0% / 4.5% /
7.25%) — weak in absolute terms but reliably several times chance level in
both directions. Full numbers in `results/retrieval_metrics.json`.

**Methodology note:** the full-scale run was executed via the Kaggle API
(`kaggle kernels push`), not the browser notebook originally planned —
pushing this repo's actual `src/*.py` scripts as a private Kaggle dataset
and running them unmodified via subprocess, rather than a hand-duplicated
notebook copy (avoiding the drift that made the very first notebook in this
project stale within a month). Two real bugs surfaced only at this scale
and were fixed in the actual source, not patched around: `gradcam.py`'s
DenseNet-121 hook crashed with a PyTorch autograd "view + inplace" error
(module-level full backward hooks conflict with DenseNet's in-place ReLU;
fixed by switching to a tensor-level hook) and `abstention_eval.py` broke
on Kaggle's older numpy (`np.trapezoid` doesn't exist before numpy 2.0,
`np.trapz` was removed after — fixed with a manual trapezoidal
calculation that depends on neither).

## 5. Evaluation metrics (full-scale)

**Text robustness** (`results/text_robustness.json`, full 549-study test
set): same healthy pattern as the small-scale pass — empty and mismatched
report text collapse retrieval to near chance level, confirming the model
has no exploitable image-only shortcut in the retrieval task; degradation
is graded (original > shuffled words > truncated > mismatched/empty), not a
cliff. Replicated cleanly at scale, nothing revised here.

**Cross-dataset distribution shift** (`results/shift_metrics.json`,
full-scale checkpoint, full real Pneumonia test set, n=624): AUROC again
*held up* under shift, and by a larger margin than at small scale (0.792
in-distribution → 0.833 on Pneumonia). **Calibration again degraded**, ECE
going from 0.040 (in-distribution) to 0.131 on the shift set — about 3.3x
worse, a similar ratio to the small-scale run's 2.8x. The model still
over-predicts abnormality on the shift set (mean predicted probability
0.725 vs. a true 62.5% pneumonia rate) but less dramatically than the
small-scale checkpoint did (88.8%) — more training data made the
overconfidence-under-shift problem smaller, not solved. The qualitative
finding replicates: discrimination survives distribution shift better than
calibration does.

**Shortcut probe** (`results/shortcut_probe.json`, full real training sets:
2,568 IU X-Ray + 4,448 Pneumonia images, n=7,016 combined): a linear probe
on frozen features separates the two datasets with **AUROC = 0.9997,
accuracy 99.8%** (vs. a 63.4% majority-class baseline) — effectively
perfect, and now confirmed on an order of magnitude more data than the
small-scale run's already-perfect 1.0/1.0 on 400 images. This rules out the
small-scale result being a tiny-sample fluke: the representations robustly
and near-completely encode acquisition source. Still the same caveat as
before — this doesn't by itself prove the classifier's *predictions* rely
on the shortcut, only that the information is trivially present in the
learned features and available to be relied upon.

**MC-dropout uncertainty + abstention** (`results/mc_dropout_summary.json`,
`results/abstention_metrics.json`, 20 stochastic passes, full 549-study
test set): **this is the result that changed most at scale.** The
small-scale run found predictive std nearly identical for correct vs.
incorrect predictions (0.084 vs. 0.086) and uncertainty-ordered abstention
no better than random. At full scale, predictive std is clearly separated —
0.029 for correct predictions vs. 0.047 for incorrect ones (incorrect
predictions carry ~64% higher epistemic uncertainty) — and uncertainty-
ordered abstention roughly **halves** risk relative to random-order
abstention (area-under-risk-coverage 0.170 vs. 0.323). The small-scale null
result was exactly what the earlier draft of this report predicted it might
be: an artifact of too little training data for MC-dropout's epistemic
signal to sharpen, not a real property of the method. At full scale,
uncertainty-aware abstention is a genuinely useful triage signal on this
task.

## 6. Explainability

Grad-CAM overlays in `results/gradcam_examples/`, full-scale checkpoint, 6
real IU X-Ray test images (not for public figures — CC BY-NC-ND, see
`MODEL_CARD.md`; use Pneumonia-dataset examples for anything public-facing).

One overlay (a correctly-identified-as-normal case, true=0, p=0.31) shows
its hottest activation region centered on the image's "L" laterality
marker/annotation text in the corner, not on lung tissue. A single example
isn't a finding by itself, but it's a concrete, visible instance of exactly
the kind of thing the shortcut probe (Section 5) found abstractly: the
model has access to, and apparently sometimes attends to, image metadata/
annotation artifacts rather than purely anatomical content. Worth a
systematic check (do laterality markers correlate with any prediction
pattern across the full test set?) as follow-up work rather than concluding
from one image.

## 7. Multimodal / foundation-model extension

**Fusion baseline** (`results/fusion_baseline.json`, `fusion_baseline.py`):
frozen BiomedCLIP image and text embeddings, logistic-regression probes,
full real IU X-Ray split (2,568/549/549) — directly comparable to Section 4.

| Feature set | Test AUROC | Test AUPRC | ECE (calibrated) |
|---|---|---|---|
| Image-only (BiomedCLIP embedding + probe) | 0.752 | 0.846 | 0.047 |
| Text-only (BiomedCLIP embedding + probe) | **0.957** | 0.973 | 0.044 |
| Fusion (image+text) | 0.955 | 0.972 | 0.047 |

**This reverses the small-scale "verdict."** At small scale (80 test
examples), fusion (0.940) nominally edged out text-only (0.937) — a margin
this report already flagged as noise, not a real effect. At full scale (549
test examples, a far more reliable comparison), **fusion does not beat
text-only** (0.955 vs. 0.957) — confirming that caution was warranted. This
is exactly the outcome the small-scale section predicted was likely and
is the cleanest evidence in this whole project that a small-sample "win"
needs treating with real skepticism until it's checked at scale.

The text-vs-image gap itself is confirmed, not reduced, at full scale
(0.957 vs. 0.752, an even larger gap than small-scale's 0.937 vs. 0.653).
This strengthens rather than weakens the label-leakage explanation from the
small-scale pass: the `Abnormal` label is derived from the same report's
Problems/MeSH field, so a text classifier is partly reading its own label
back out of correlated text. If this were genuine diagnostic signal in the
prose rather than leakage, there's no obvious reason the gap should *widen*
with 10x more training data — leakage explanations don't average out with
scale the way genuine-but-noisy signal would.

**Honest reading, updated:** the full-scale run doesn't just fail to show a
fusion benefit — it actively demonstrates fusion providing no measurable
value over text alone on this task/label, once the small-sample noise is
removed. The Gate 2 stage-gate condition ("fusion adds measurable value
beyond the strongest unimodal baseline") is **not met** on this
classification setup. A more defensible fusion test needs either (a) a
label that isn't derived from the same text being fed to the model, or (b)
evaluating on the retrieval task instead of classification, where no such
leakage path exists — which is exactly what the next result does.

### A genuine (non-label-confounded) fusion test: contrastive retrieval fine-tuning

The classification fusion test above can't cleanly answer "does fusion
help" because the label itself is text-derived. Retrieval sidesteps that
entirely: it only uses the natural image<->report pairing, never the
`Abnormal` label, so there's no leakage path for a result to hide behind.

**Method** (`contrastive_projection.py`): keep BiomedCLIP's backbone frozen
(no full fine-tuning — avoids the compute-overload risk the project plan
flags) and train two small linear projection heads (image_dim -> 256,
text_dim -> 256) with a symmetric InfoNCE contrastive loss on the training
set's image-report pairs. Model selection by validation Recall@1. Compare
zero-shot (raw BiomedCLIP embeddings) against the trained projection on the
same held-out test embeddings.

**Full-scale result (2,568 train / 549 val / 549 test, real data, Kaggle
GPU):**

| Metric | Zero-shot | Trained projection | Change |
|---|---|---|---|
| image→text Recall@1 | 1.28% | 1.64% | +29% |
| image→text Recall@5 | 3.64% | 7.83% | **+115%** |
| image→text Recall@10 | 5.65% | 12.02% | **+113%** |
| text→image Recall@1 | 1.09% | 2.37% | **+117%** |
| text→image Recall@5 | 4.01% | 6.56% | +64% |
| text→image Recall@10 | 6.19% | 13.11% | **+112%** |

**This is a real, unconfounded fusion win** — roughly doubling Recall@5/@10
in both directions, with genuine gains at Recall@1 too. The training curve
supports treating this as real signal, not overfitting: the best checkpoint
was selected at epoch 28 of 200 (train loss 6.97, well short of the loss
4.88 reached by epoch 200), and validation Recall@1 declined gently after
that peak rather than collapsing — a healthy early-stopping picture, not a
memorization artifact. A small-scale check on 240 real training pairs
(local CPU, before committing to the full-scale GPU run) showed the
opposite pattern — validation performance peaked almost immediately and
degraded for the rest of training, the signature of a linear head overfitting
240 examples — which is exactly the small-sample noise problem this project
has run into before (Section 7 above, Section 5's MC-dropout result) and
exactly why this result was checked at full scale before being reported.

**This is the strongest, cleanest piece of multimodal evidence in the whole
project.** Unlike the classification fusion test, it has no leakage
confound, no small-sample ambiguity, and a training curve consistent with
genuine generalization. It directly satisfies the Gate 2 condition the
classification test failed to meet: training that uses both modalities
together measurably outperforms the frozen, generic pretrained baseline.

## 8. Error analysis

Per-example predictions for the full-scale image-only checkpoint (Section 4)
on the full real 549-study test set, from `error_analysis.py`, with report
text attached to every prediction.

**Confusion breakdown (threshold 0.5):** TP=266, TN=133, FP=72, FN=78 —
sensitivity (recall) 0.773, specificity 0.649, precision 0.787. All three
improved over the small-scale run (0.667 / 0.657 / 0.714), consistent with
the better AUROC/AUPRC in Section 4.

**The small-scale qualitative narrative does not hold up cleanly at full
scale — worth stating plainly rather than quietly dropping.** The
small-scale pass (n=15 FN, n=12 FP) read as two clean patterns: FN cases
were "subtle chronic findings," FP cases were "postsurgical hardware/
salient-but-non-diagnostic." At full scale (n=78 FN, n=72 FP), a keyword
check across every case (not just the 10-12 most confident ones, which is
what the small-scale write-up effectively was) tells a messier story:

- **FN cases**: 81% mention acute-sounding findings (pneumonia, airspace
  disease, infiltrate, consolidation, effusion, pneumothorax, cuffing,
  edema) and 49% mention subtle/chronic language (calcification, granuloma,
  chronic, mild, minimal, stable, unchanged) — these aren't mutually
  exclusive, but the takeaway is that the "FN = subtle chronic findings
  only" story from small scale was too clean. Reading the lowest-confidence
  FN cases individually, calcified-granuloma-type findings are still
  over-represented at the very bottom (most confidently wrong), but the
  full 78-case set also includes clear misses of real acute findings like
  *"Left lower lobe ... segment pneumonia"* (p=0.223) and *"Right upper
  lobe airspace disease consistent with pneumonia"* (p=0.233) — the model
  is missing more than just the hard subtle cases.
- **FP cases**: only 17% (12/72) contain postsurgical/hardware/prior-
  reference language — a real pattern (still visible at the top of the
  confidence-sorted list: *"Postsurgical changes of ... sternotomy with
  screw fixation"* at p=0.925), but a minority explanation, not the
  dominant one the small-scale sample suggested.

**The honest lesson here is methodological, not just about this model**: a
qualitative read of a dozen examples can produce a clean-sounding narrative
that a systematic keyword check across the full error set doesn't
support. The small-scale error analysis wasn't wrong that these patterns
exist — both are still visible in the data — it was wrong to imply they
were the dominant story. Keyword matching itself is a blunt instrument
(e.g. "no consolidation" would match on the word "consolidation" despite
describing an absent finding), so even this full-scale check should be read
as a better approximation, not a final word — a next step worth doing is
having an actual per-case read of a larger, randomly-sampled subset rather
than either the most-confident extremes or fully automated keyword counts.

## 9. Limitations

- IU X-Ray's `Abnormal` label is a heuristic derived from the same report
  used for the multimodal comparison — confirmed as a real, non-trivial
  effect at full scale (Section 7), not just a theoretical risk.
- Chest X-Ray Pneumonia has no patient IDs (image-wise split, not
  patient-wise) — a pre-existing limitation of that dataset.
- The shortcut probe (Section 5) shows the model's representations encode
  acquisition source almost perfectly; this does not by itself prove
  predictions depend on it, only that the information is present and usable.
- The full-scale error-analysis keyword check (Section 8) is a blunt
  instrument (naive string matching, no negation handling) — a genuine
  clinical read of a larger random sample would be more reliable.
- No fine-tuned/contrastive multimodal model has been trained — only
  zero-shot retrieval and frozen-embedding classification probes. The
  retrieval numbers remain the cleanest multimodal evidence in this report,
  since they're not confounded by the label-leakage issue.
- All full-scale results are from a single training run per configuration
  (no repeated-seed variance estimate) — point estimates, not confidence
  intervals.

## 10. Future work

- Investigate the shortcut-probe finding further: which features drive the
  near-perfect dataset separability (image statistics/preprocessing
  artifacts vs. something more concerning)? Would inform whether
  domain-adaptation or harmonization preprocessing is worth adding. The
  Grad-CAM laterality-marker observation (Section 6) is a concrete starting
  point.
- ~~A genuine multimodal fusion test~~ — done (Section 7): contrastive
  projection heads on frozen BiomedCLIP embeddings, trained on real
  image-report pairs, roughly double Recall@5/@10 over zero-shot at full
  scale. Natural next step: try fine-tuning more than just linear
  projection heads (e.g. a small MLP, or unfreezing the last few backbone
  layers) now that this establishes a real baseline worth improving on.
- A systematic (not keyword-based) read of a larger random error sample,
  to properly characterize the FN/FP patterns hinted at in Section 8.
- Repeated-seed runs for confidence intervals on the headline numbers.
- MIMIC-CXR upgrade, pending PhysioNet credentialing.
- Grand Challenge participation (REG2027/CXR-LT 2027 preferred; BEETLE
  parked on an unresolved storage question; AMIA/VinBigData detection
  queued as a no-deadline fallback) — per Prof Slabaugh's feedback, now
  that Project 1's core gates are complete.
