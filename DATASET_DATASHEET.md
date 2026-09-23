# Dataset Datasheet

## Primary: IU X-Ray / Open-i (Indiana University Chest X-Ray Collection)

- **Source:** NIH NLM Open-i, mirrored on Kaggle as
  [`raddar/chest-xrays-indiana-university`](https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university)
- **License:** **CC BY-NC-ND 4.0** (Attribution-NonCommercial-NoDerivatives),
  as stated on the Kaggle listing — verified 2026-08-12. This is more
  restrictive than initially assumed: NonCommercial is fine for a
  non-commercial research/portfolio project, but **NoDerivatives means
  modified/processed versions of the images (resized crops, Grad-CAM
  overlays, etc.) should not be published publicly** without the
  rightsholder's permission. Practical effect on this
  repo: raw and derived images stay out of git (already the case, via
  `.gitignore`), and any Grad-CAM/qualitative figures used in the public
  README or technical report should come from the Pneumonia dataset (CC BY
  4.0, no ND restriction) or be described/summarized rather than shown as
  image files, until this is resolved.
- **Size:** 7,470 frontal/lateral images total (13.2GB full download); one
  frontal image per study is kept after filtering (3,666 studies with both a
  usable report and a derivable label, verified from the actual downloaded
  metadata on 2026-08-12: 2,568 train / 549 val / 549 test).
- **Text:** free-text radiology reports with `findings` and `impression`
  sections, concatenated as `report_text`.
- **Label:** `Abnormal` (0/1) is a **weak, heuristic label** derived from the
  `Problems`/MeSH field, not a clinician adjudication — see the label-leakage
  discussion in `MODEL_CARD.md`.
- **Splits:** by study `uid` (`src/prepare_iuxray_csv.py`), which functions as
  a patient-level split for this corpus since each `uid` is one study/patient.
- **Preprocessing here:** resize to model input size, ImageNet normalization
  (classification path) or the model-specific preprocessing bundled with
  BiomedCLIP (retrieval path); random flip + small rotation as train-time
  augmentation for classification only.
- **Ethics:** de-identified public research dataset; no additional PHI
  handling required beyond what NLM/Open-i already performed.

## Secondary: Chest X-Ray Pneumonia (cross-dataset shift test only)

- **Source:** Kermany, D. et al. (2018); mirrored on Kaggle by Paul Mooney —
  https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia
- **License:** CC BY 4.0
- **Population & acquisition:** Pediatric patients (ages 1-5), anterior-posterior
  chest X-rays, Guangzhou Women and Children's Medical Center — a different
  hospital, patient population, and acquisition setting than IU X-Ray, which
  is exactly why it's used as the shift-test set.
- **Size:** 5,863 images, 2 classes (NORMAL, PNEUMONIA), no report text.
- **Splits:** the dataset's own `val/` folder has only 16 images, too small to
  validate on, so `src/prepare_csv.py` pools `train/` + `val/` and creates a
  fresh stratified 85/15 train/val split; the original `test/` folder is left
  untouched as the held-out test set.
- **Known limitation:** no patient IDs are published, so this is an
  image-wise split, not a patient-wise split.

## Planned: MIMIC-CXR

Not yet accessible — requires PhysioNet credentialing. See `data/README.md`
for the concrete steps (account, CITI training, application, DUA signature).
Would replace IU X-Ray as the primary dataset if/when access is granted, given
its far larger scale and status as the field-standard benchmark.
