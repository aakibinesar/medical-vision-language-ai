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

## 4. Baseline models — preliminary results

**Caveat: these are from a small, real (non-synthetic) 240/80/80-study local
subsample of IU X-Ray (CPU-trained, ResNet-18, 160px), used to validate the
full pipeline end-to-end on real data while a laptop with only 2GB VRAM
can't run a full-scale training job.** The full-scale run (2,568 train
studies, 224px, DenseNet-121, on Kaggle's free GPU via
`notebooks/kaggle_baseline_training.ipynb`) is what should anchor the final
reported numbers — treat everything below as a pipeline-correctness proof,
not the headline result.

| Metric | Value |
|---|---|
| Test AUROC | 0.749 |
| Test AUPRC | 0.810 |
| ECE before calibration | 0.137 |
| ECE after temperature scaling | 0.094 |

Training itself overfit after ~4-5 epochs (val AUROC peaked at 0.721 at
epoch 4, then declined to 0.562 by epoch 8 while train loss kept falling) —
expected and worth reporting honestly given only 240 training images; the
full-scale run should show a cleaner curve.

**Zero-shot BiomedCLIP retrieval** (400-study real IU X-Ray subsample, no
training): image→text Recall@1/5/10 = 2.0% / 4.5% / 7.25% (chance = 0.25% /
1.25% / 2.5% with 400 candidates) — 3-8x better than chance but weak in
absolute terms. Full numbers in `results/retrieval_metrics.json`.

## 5. Evaluation metrics

**Text robustness** (`results/text_robustness.json`): empty and mismatched
report text collapse retrieval to almost exactly chance level, confirming
the model has no exploitable image-only shortcut; degradation is graded
(original > shuffled words > truncated > mismatched/empty), not a cliff.

**Cross-dataset distribution shift** (`results/shift_metrics.json` — same
small local checkpoint, evaluated on the full real Pneumonia test set,
n=624): AUROC actually *held up* under shift (0.749 in-distribution →
0.771 on Pneumonia), but **calibration broke down badly**: ECE went from
0.094 (in-distribution, post temperature-scaling) to 0.263 on the shift
set, with the model averaging 88.8% predicted-abnormal probability against
a true 62.5% pneumonia rate — systematically overconfident under shift even
though its ranking of cases stayed reasonable. Discrimination and
calibration are not the same kind of robustness, and this result is exactly
why the project plan treats them as separate mandatory evaluation blocks.

**Shortcut probe** (`results/shortcut_probe.json`, 200 real images per
dataset): a linear probe on the trained model's frozen penultimate features
separates IU X-Ray from Pneumonia-dataset images with **AUROC = 1.0** —
perfect separability. This is a real shortcut-learning flag: the
representations strongly encode acquisition source (hospital/scanner/
preprocessing), consistent with well-documented findings in the medical
imaging robustness literature that classifiers often pick up "which
institution" signal alongside or instead of pathology signal. It also
tempers the shift result above — some of the preserved AUROC under shift
could be aided by the model implicitly recognizing "this is a different
kind of scan" rather than purely reasoning about pathology.

**MC-dropout uncertainty + abstention** (`results/mc_dropout_summary.json`,
`results/abstention_metrics.json`, 20 stochastic passes, real 80-study test
set): predictive std was nearly identical between correct and incorrect
predictions (0.084 vs 0.086), and uncertainty-ordered abstention performed
*no better than random-order abstention* (area-under-risk-coverage 0.3117
vs 0.3117). Honest negative result at this scale: with only 240 training
images, the dropout-based uncertainty estimate isn't yet informative enough
to usefully triage predictions. Worth re-running at full scale (more
training data typically sharpens MC-dropout's epistemic signal) before
drawing a final conclusion.

## 6. Explainability

Grad-CAM overlays in `results/gradcam_examples/` (Pneumonia-dataset examples
only for any public figures — IU X-Ray is CC BY-NC-ND, see `MODEL_CARD.md`).

## 7. Multimodal / foundation-model extension

**Fusion baseline** (`results/fusion_baseline.json`, `fusion_baseline.py`):
frozen BiomedCLIP image and text embeddings, logistic-regression probes,
same 240/80/80 real IU X-Ray split as Section 4, so this is directly
comparable to the CNN baseline.

| Feature set | Test AUROC | Test AUPRC | ECE (calibrated) |
|---|---|---|---|
| Image-only (BiomedCLIP embedding + probe) | 0.653 | 0.731 | 0.075 |
| Text-only (BiomedCLIP embedding + probe) | 0.937 | 0.946 | 0.120 |
| Fusion (image+text) | 0.940 | 0.950 | 0.100 |

**Gate 2's literal bar — "fusion beats the best unimodal baseline" — is
technically met (0.940 vs. 0.937), but the margin is 0.003 AUROC on an
80-example test set: not distinguishable from noise, and should not be
reported as "fusion works."**

The finding that actually matters here is text-only (0.937) dramatically
outperforming image-only (0.653). This is **not** strong evidence that
report text carries far more diagnostic signal than the image — it's the
label-leakage risk flagged in `MODEL_CARD.md` showing up empirically: the
`Abnormal` label is derived from the same report's Problems/MeSH field, so
a text classifier is partly reading its own label back out of correlated
text, not doing independent clinical reasoning. The image-only probe
(0.653) also underperforms the separately fine-tuned end-to-end ResNet-18
CNN baseline (0.749, Section 4) — expected, since BiomedCLIP's image tower
here is frozen and generic, not fine-tuned on this task, unlike the CNN.

**Honest reading:** this baseline doesn't yet demonstrate that multimodal
fusion adds real value beyond what a leaky text-derived label already gives
away. A more defensible fusion test would need either (a) a label that
isn't derived from the same text being fed to the model, or (b) evaluating
on the retrieval task instead of classification, where no such leakage path
exists (see Section 4's zero-shot retrieval numbers, which don't have this
confound). Both are queued as future work rather than re-run now, per the
"one dataset, controlled scope" sequencing this project has followed
throughout.

## 8. Error analysis

(TODO.)

## 9. Limitations

- All classification numbers to date are from a small local CPU subsample,
  not the full training set — see the caveat in Section 4.
- IU X-Ray's `Abnormal` label is a heuristic derived from the same report
  used for the multimodal comparison — see the label-leakage discussion in
  `MODEL_CARD.md`.
- Chest X-Ray Pneumonia has no patient IDs (image-wise split, not
  patient-wise) — a pre-existing limitation of that dataset.

## 10. Future work

- Run the full-scale training job on Kaggle (Gate 1 completion) and re-run
  every Gate 3 evaluation (shift, shortcut, MC-dropout/abstention) against
  it — small-scale results above are directionally interesting but need
  confirming at scale, especially the null abstention result.
- Investigate the shortcut-probe finding further: which features drive the
  perfect dataset separability (image statistics/preprocessing artifacts vs.
  something more concerning)? Would inform whether domain-adaptation or
  harmonization preprocessing is worth adding.
- Re-test multimodal fusion with a label that isn't text-derived (removes
  the leakage confound found in Section 7), or lean on the retrieval task
  instead of classification, where the confound doesn't apply.
- MIMIC-CXR upgrade, pending PhysioNet credentialing.
