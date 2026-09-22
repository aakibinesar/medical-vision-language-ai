# TrustMed-VLM: Trustworthy Medical Vision-Language AI Under Distribution Shift

**Novelty target (from the Project 1 red-team assessment):** a systematic
comparison of image-only and medical vision-language models under limited
labels, missing or corrupted text, calibration error, shortcut dependence,
and cross-environment distribution shift, including uncertainty-aware
abstention. The result is not "we fine-tuned a VLM on medical images" — it's
the reliability/robustness comparison itself.

## Key findings at a glance

- **A full pipeline audit caught three real bugs, including one with a
  physical explanation for an earlier finding.** Training used
  `RandomHorizontalFlip`, inappropriate for chest X-rays (anatomy isn't
  left-right symmetric, and it mirrors laterality markers into nonsense) —
  plausibly why Grad-CAM was found attending to an "L" marker. Fixing it and
  fully retraining left classification performance unchanged within noise
  but *improved calibration stability* (seed std roughly halved) and changed
  that Grad-CAM case's behavior, though not cleanly (Section 6). A
  confusion-matrix labeling bug and an unseeded abstention baseline were
  fixed at the same time (Sections 4-5).
- **Discrimination and calibration are consistently different axes of
  reliability** — the recurring theme of this report. AUROC survives
  distribution shift and is stable across training seeds; calibration
  degrades under shift and is more seed-variable than discrimination,
  though less so after the pipeline fix above (Sections 4-5).
- **The genuine fusion win is in retrieval, not classification.** A
  contrastive-projection model trained on real image-report pairs roughly
  doubles retrieval Recall@5/@10 over zero-shot BiomedCLIP, confirmed
  across 3 seeds (Section 7). The classification "fusion" test looked
  promising at small scale but was noise — at full scale it doesn't beat a
  text-only baseline confounded by label leakage (Section 7).
- **MC-dropout uncertainty is scale-dependent**: useless at small scale
  (null result), genuinely informative at full scale — uncertainty-ordered
  abstention roughly halves risk vs. random (Section 5).
- **A near-perfect shortcut signal exists**: a linear probe tells IU X-Ray
  and Pneumonia images apart with AUROC 0.9997 from frozen features alone
  (Section 5), and one Grad-CAM example shows the model attending to an
  image annotation marker rather than anatomy (Section 6).
- **Small-scale qualitative read do not always survive full-scale
  scrutiny** — the clearest lesson of the whole project. This showed up at
  least three times: the fusion classification "win" (Section 7), the
  error-analysis FP/FN narrative (Section 8), and the single-seed ECE
  headline number (Section 4). None were wrong to notice; all were
  overstated until checked at scale or repeated over seeds.

## 1. Motivation

Clinical decision support built on medical imaging has to work outside the
exact distribution it was validated on — different hospitals, scanners,
patient populations, and label conventions than whatever produced the
training set. A model that scores well on held-out data from its own
source but silently becomes overconfident or unreliable elsewhere is a
worse deployment risk than one that is honestly mediocre everywhere,
because the failure is invisible until it causes harm. This project treats
that reliability question as the actual research object, not an
afterthought bolted onto an accuracy number: given a chest X-ray
classification task with paired radiology reports, does combining image
and text carry real information beyond either alone, and does whatever
model results stay trustworthy — calibrated, robust to shift, honest about
its own uncertainty — outside the exact setting it was built in? The
project is deliberately structured so accuracy is necessary but not
sufficient: a result only counts once it has been checked for calibration,
distribution shift, shortcut dependence, and (where relevant) whether it
survives being re-run at a larger scale or across multiple seeds.

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

**Note: numbers below reflect a corrected training pipeline.** A later full
pipeline audit found that `dataset.py`'s training augmentation included
`RandomHorizontalFlip`, inappropriate for chest X-rays — anatomy isn't
left-right symmetric (heart, aortic arch), and flipping mirrors embedded
laterality markers into unreadable nonsense, plausibly explaining the
Grad-CAM finding in Section 6. Every checkpoint below was retrained without
it; classification performance changed only within seed-to-seed noise, and
calibration got *more* seed-stable (see the repeated-seed table below). A
hard-coded confusion-matrix labeling bug was fixed at the same time. See
`kaggle/recheck-augmentation/` and `kaggle/no-flip-full-ci/`.

**These are the headline numbers**, from the full-scale run: DenseNet-121,
224px, on the complete real IU X-Ray split (2,568 train / 549 val / 549
test), run via the Kaggle API (see the note on methodology at the end of
this section). An earlier small-scale pass (240/80/80 studies, ResNet-18,
160px, local CPU) was used to validate the pipeline before committing to
full-scale compute; those numbers are kept below each result for
comparison, since several of them changed meaningfully at scale — that
delta is itself informative about which small-scale findings were real
signal versus small-sample noise.

| Metric | Full-scale (seed 42, headline) | Small-scale (pipeline check) |
|---|---|---|
| Test AUROC | **0.785** | 0.749 |
| Test AUPRC | **0.859** | 0.810 |
| ECE before calibration | **0.069** | 0.137 |
| ECE after temperature scaling | **0.071** | 0.094 |

More training data improved both discrimination and calibration relative to
the small-scale check, as expected. (Pre-fix, with the flip, this run read
AUROC 0.792 / AUPRC 0.865 / ECE 0.052→0.040 — essentially the same result
within noise; see the repeated-seed table below for why the single-seed
calibration number moved.) Full history in `outputs_full_no_flip/history.json`.

**Repeated-seed confidence interval** (`results/metrics_seed_ci.json`,
seeds 42/43/44, identical hyperparameters, only the random seed differs):

| Metric | Mean ± std (n=3) | Per-seed values (42 / 43 / 44) |
|---|---|---|
| Test AUROC | 0.772 ± 0.011 | 0.785 / 0.766 / 0.766 |
| Test AUPRC | 0.855 ± 0.008 | 0.859 / 0.860 / 0.847 |
| ECE after calibration | 0.072 ± 0.022 | 0.071 / 0.095 / 0.051 |

**Calibration is still more seed-sensitive than discrimination, but the
pipeline fix (see the note above the first table) noticeably tightened it.**
AUROC and AUPRC remain tight and stable across seeds (std around 1%
relative). ECE after calibration now ranges 0.051–0.095 (std 0.022) —
compared to 0.040–0.130 (std 0.045) before removing `RandomHorizontalFlip`,
roughly half the spread on a similar mean. Unlike the pre-fix run, the
best-AUROC seed (42, 0.785) is no longer also the best-calibrated (that's
now seed 44, ECE 0.051) — a hint of a real discrimination/calibration
trade-off, though with n=3 this is far too little data to treat as
established. The honest calibration number to report going forward is
0.072 ± 0.022.

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
no-flip full-scale checkpoint, full real Pneumonia test set, n=624): AUROC
again *held up* under shift, and by a larger margin than at small scale
(0.785 in-distribution → 0.847 on Pneumonia). **Calibration again
degraded**, ECE going from 0.071 (in-distribution) to 0.083 on the shift
set — a much more modest ~17% relative increase than the pre-fix run's
~3.3x (0.040→0.131). The model still over-predicts abnormality on the
shift set (mean predicted probability 0.602 vs. a true 62.5% pneumonia
rate — now close to calibrated on this axis, versus 0.725 pre-fix) but the
qualitative finding still replicates: discrimination survives distribution
shift better than calibration does, just by a smaller margin now that
training-time noise from the flip is gone.

**Shortcut probe** (`results/shortcut_probe.json`, full real training sets:
2,568 IU X-Ray + 4,448 Pneumonia images, n=7,016 combined): a linear probe
on frozen features separates the two datasets with **AUROC = 0.99995,
accuracy 99.6%** (vs. a 63.4% majority-class baseline) — effectively
perfect, and now confirmed on an order of magnitude more data than the
small-scale run's already-perfect 1.0/1.0 on 400 images, and unaffected by
the augmentation fix (as expected — it probes frozen representations, not
augmentation-sensitive predictions). This rules out the small-scale result
being a tiny-sample fluke: the representations robustly and
near-completely encode acquisition source. Still the same caveat as
before — this doesn't by itself prove the classifier's *predictions* rely
on the shortcut, only that the information is trivially present in the
learned features and available to be relied upon.

**MC-dropout uncertainty + abstention** (`results/mc_dropout_summary.json`,
`results/abstention_metrics.json`, 20 stochastic passes, full 549-study
test set, no-flip checkpoint): **this is the result that changed most at
scale.** The small-scale run found predictive std nearly identical for
correct vs. incorrect predictions (0.084 vs. 0.086) and uncertainty-ordered
abstention no better than random. At full scale, predictive std is clearly
separated — 0.035 for correct predictions vs. 0.051 for incorrect ones
(incorrect predictions carry ~44% higher epistemic uncertainty) — and
uncertainty-ordered abstention scores clearly lower (better) than
random-order abstention (area-under-risk-coverage 0.157 vs. 0.264, the
latter now averaged over 100 random permutations rather than a single
arbitrary draw — see the pipeline-fix note). The small-scale null result
was exactly what the earlier draft of this report predicted it might be: an
artifact of too little training data for MC-dropout's epistemic signal to
sharpen, not a real property of the method. At full scale, uncertainty-aware
abstention is a genuinely useful triage signal on this task.

**Repeated-seed confirmation for all four diagnostics above**
(`results/gate3_seed_ci.json`, seeds 42/43/44, no-flip checkpoints
throughout — the seed-42 main checkpoint reused from Section 4, 5 new
checkpoints trained for the rest). Every qualitative finding still holds
up cleanly across seeds after the pipeline fix, and calibration is again
the metric that improved most:

| Metric | Mean ± std (n=3) |
|---|---|
| Shift AUROC | 0.863 ± 0.022 (all seeds above their own in-distribution AUROC) |
| Shift ECE | 0.086 ± 0.014 (down from 0.117 ± 0.026 pre-fix, still above in-distribution ECE) |
| Shortcut probe AUROC | 0.99992 ± 0.00003 (essentially seed-invariant, unaffected by the fix) |
| MC-dropout std, correct predictions | 0.036 ± 0.005 |
| MC-dropout std, incorrect predictions | 0.049 ± 0.004 (higher than correct, every seed) |
| Abstention AUC-risk, uncertainty-ordered | 0.166 ± 0.008 |
| Abstention AUC-risk, random-ordered | 0.261 ± 0.005 (worse than uncertainty-ordered, every seed; now itself averaged over 100 permutations per seed rather than one arbitrary draw) |

The shortcut probe remains essentially seed-invariant — this is a property
of the *data*, not an artifact of one particular trained model, and
unaffected by the augmentation fix as expected. MC-dropout's
correct-vs-incorrect std gap and abstention's uncertainty-vs-random gap
both hold in the same direction for all three seeds, with no overlap
between the two conditions' ranges in either case — a real, repeatable
effect, not a coincidence of seed 42. As in Section 4, the clearest
before/after change from the pipeline fix is calibration-related: shift ECE
std dropped from 0.026 to 0.014, mirroring the tighter Gate 1 calibration
spread — removing the flip made calibration more consistent across seeds
generally, not just for the in-distribution result.

## 6. Explainability

Grad-CAM overlays in `results/gradcam_examples/`, no-flip full-scale
checkpoint, 6 real IU X-Ray test images (not for public figures — CC
BY-NC-ND, see `MODEL_CARD.md`; use Pneumonia-dataset examples for anything
public-facing).

One overlay (`3089_IM-1444-1001`, a correctly-identified-as-normal case,
true=0) originally showed its hottest activation region centered sharply
and asymmetrically on the image's "L" laterality marker/annotation text,
not on lung tissue — a concrete, visible instance of exactly the kind of
thing the shortcut probe found abstractly. The pipeline audit (see the note
at the start of Section 4) identified training-time `RandomHorizontalFlip`
as a plausible cause: chest X-ray anatomy isn't left-right symmetric, and
flipping mirrors embedded laterality markers into unreadable nonsense,
which could teach a model that a marker's presence/shape — rather than its
specific content — is a usable cue.

**Direct before/after check on this exact image, no-flip checkpoint vs.
the original:** the sharp, asymmetric fixation on the "L" marker is gone.
In its place, the no-flip checkpoint shows broader, *symmetric* hot
attention across **both** shoulder/collar corners — the "SAW" marker on the
opposite side is now equally hot, not just the "L" marker. The predicted
probability for this image also shifted (0.15 → 0.44, both correctly below
the 0.5 threshold). **This is a real, honest, partial result, not a clean
fix**: the flip does appear to be at least part of the cause of the
original sharp marker-fixation, but the model still isn't attending to
lung tissue on this image — it has moved from one peripheral shortcut
pattern (asymmetric letter-fixation) to another (symmetric corner/shoulder
attention). A single example still isn't a systematic finding by itself; a
worthwhile follow-up (not yet done) would check whether this pattern holds
across more of the test set now that the augmentation is fixed.

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

**Repeated-seed confirmation (`results/contrastive_projection.json`,
seeds 42/43/44, embeddings extracted once and reused — only the projection
heads' random initialization varies per seed):**

| Metric | Zero-shot | Trained (mean ± std, n=3) |
|---|---|---|
| image→text Recall@1 | 1.28% | 1.76% ± 0.46% |
| image→text Recall@5 | 3.64% | 7.41% ± 0.42% |
| image→text Recall@10 | 5.65% | 12.26% ± 0.92% |
| text→image Recall@1 | 1.09% | 2.06% ± 0.76% |
| text→image Recall@5 | 4.01% | 6.98% ± 0.42% |
| text→image Recall@10 | 6.19% | 11.84% ± 1.09% |

The single-seed result above (seed 42) wasn't a lucky draw: across three
seeds the standard deviation is small (0.4-1.1 percentage points) relative
to the roughly 2x effect size, and the zero-shot baseline sits well outside
the trained mean's range for every single metric. This is a genuinely
robust result, not a point estimate that happened to land well.

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

- The RandomHorizontalFlip fix (Section 4) changed one Grad-CAM case's
  attention pattern but did not make it attend to lung tissue — it moved
  from one peripheral shortcut (asymmetric marker-fixation) to another
  (symmetric shoulder/corner attention). The underlying shortcut-learning
  tendency itself is not resolved, only one specific manifestation of it.
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
- The contrastive projection (Section 7) only trains small linear heads on
  a fully frozen backbone — it doesn't establish how much further gains
  might come from fine-tuning more of the network, just that even this
  minimal training already helps.
- Repeated-seed confidence intervals (n=3 seeds each) now exist for the
  Gate 1 classification headline numbers (Section 4), the contrastive
  projection fusion result (Section 7), and all four Gate 3 diagnostics —
  shift, shortcut probe, MC-dropout, and abstention (Section 5). n=3 is
  enough to catch the calibration instability in Section 4 and to confirm
  the Section 5 findings are seed-robust, but it's still a small-N estimate
  of variance — a std from 3 samples is itself noisy; don't over-interpret
  the exact std values, only the qualitative patterns.
- The classification fusion baseline (confounded by label leakage) and the
  full-scale error-analysis keyword check are still single-run against the
  seed-42 checkpoint only — not extended to repeated seeds, since both are
  already flagged as unreliable/blunt-instrument results on other grounds
  (Sections 7 and 8) where a tighter variance estimate wouldn't change the
  conclusion.

## 10. Future work

- Investigate the shortcut-probe finding further: which features drive the
  near-perfect dataset separability (image statistics/preprocessing
  artifacts vs. something more concerning)? Would inform whether
  domain-adaptation or harmonization preprocessing is worth adding.
- ~~Investigate the Grad-CAM laterality-marker observation~~ — partially
  done (Section 6): traced to `RandomHorizontalFlip`, fixed, and confirmed
  the specific marker-fixation behavior changed. Natural extension: a
  systematic check across more of the test set (not just one image) for
  whether peripheral/shoulder-region shortcut attention is now the more
  general pattern replacing the marker-specific one.
- ~~A genuine multimodal fusion test~~ — done (Section 7): contrastive
  projection heads on frozen BiomedCLIP embeddings, trained on real
  image-report pairs, roughly double Recall@5/@10 over zero-shot at full
  scale. Natural next step: try fine-tuning more than just linear
  projection heads (e.g. a small MLP, or unfreezing the last few backbone
  layers) now that this establishes a real baseline worth improving on.
- A systematic (not keyword-based) read of a larger random error sample,
  to properly characterize the FN/FP patterns hinted at in Section 8.
- ~~Repeated-seed runs for confidence intervals on the headline numbers~~ —
  done for Gate 1 classification, Gate 2 fusion, and all four Gate 3
  diagnostics (n=3 each). Natural extension: n=5+ for a less noisy std
  estimate, particularly for Section 4's calibration number where n=3
  already showed real variance worth pinning down more precisely.
- MIMIC-CXR upgrade, pending PhysioNet credentialing.
- Grand Challenge participation (REG2027/CXR-LT 2027 preferred; BEETLE
  parked on an unresolved storage question; AMIA/VinBigData detection
  queued as a no-deadline fallback) — per Prof Slabaugh's feedback, now
  that Project 1's core gates are complete.
