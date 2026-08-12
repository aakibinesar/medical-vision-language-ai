# Data

This folder is intentionally empty in git. No medical images are committed to the repository.

Per the Project 1 (TrustMed-VLM) plan, this repo now uses **two** datasets
with different roles:

| Dataset | Role | Has paired text? |
|---|---|---|
| IU X-Ray / Open-i | **Primary.** Image-only baseline + VLM/retrieval comparison + text-robustness tests | Yes |
| Chest X-Ray Pneumonia | **Secondary.** Cross-dataset distribution-shift test (train on IU X-Ray, evaluate generalisation here) | No |
| MIMIC-CXR | **Planned upgrade**, pending PhysioNet credentialing (see below) | Yes |

## Primary dataset: IU X-Ray / Open-i (Indiana University Chest X-Ray Collection)

Source: NIH NLM Open-i (https://openi.nlm.nih.gov/), mirrored on Kaggle as
[`raddar/chest-xrays-indiana-university`](https://www.kaggle.com/datasets/raddar/chest-xrays-indiana-university).
~7,470 frontal/lateral chest X-rays with free-text radiology reports (findings +
impression). Check the Kaggle page's license field before any redistribution.

**Getting the data (same pattern as before):**
1. Kaggle Notebook: add "Chest X-rays (Indiana University)" via *Add Input*; it
   appears at `/kaggle/input/chest-xrays-indiana-university/`.
2. Local: `kaggle datasets download -d raddar/chest-xrays-indiana-university -p data --unzip`,
   producing `data/indiana_reports.csv`, `data/indiana_projections.csv`, and
   `data/images/images_normalized/*.png`.

**Build the CSVs:**
```bash
python src/prepare_iuxray_csv.py \
    --reports-csv data/indiana_reports.csv \
    --projections-csv data/indiana_projections.csv \
    --out-dir data_iuxray
```
This keeps one frontal image per study, joins it to the report text, derives a
weak `Abnormal` label from the `Problems`/MeSH field, and splits by `uid`
(effectively patient-level for this corpus — see `MODEL_CARD.md` for why the
label is "weak" and must be treated as noisy, not ground truth).

## Secondary dataset: Chest X-Ray Pneumonia (cross-dataset shift test)

Same as before — see the earlier section of this file's git history, or
`src/prepare_csv.py`. Its role changed: it's no longer the primary dataset,
it's the held-out generalisation check (different hospital, different label
scheme, no report text) for whatever is trained on IU X-Ray.

## Planned upgrade: MIMIC-CXR

MIMIC-CXR is the field-standard large-scale chest X-ray + report dataset
(~377k images) and would meaningfully strengthen this project, but access
requires PhysioNet credentialing — **this cannot be automated and has real
lead time, so start it now if you want it available later:**

1. Create a PhysioNet account: https://physionet.org/register/
2. Complete the required CITI "Data or Specimens Only Research" human-subjects
   training course (free, ~a few hours): https://physionet.org/about/citi-course/
3. Submit your credentialing application on PhysioNet with your institutional
   affiliation (or independent-researcher justification) and a brief research
   use case.
4. Once approved, sign the MIMIC-CXR data use agreement and request access to
   `mimic-cxr-jpg` on PhysioNet: https://physionet.org/content/mimic-cxr-jpg/
5. Approval can take anywhere from a few days to a couple of weeks. Once
   through, the same `prepare_iuxray_csv.py`-style pattern (report text +
   weak/derived labels, patient-level split via `subject_id`) will need a
   MIMIC-specific variant, since MIMIC's report sections and CSV schema differ
   from IU X-Ray's.

None of this can be done on your behalf (it requires your own identity
verification and signature on the data use agreement) — this is the one
genuinely blocking, non-automatable step in the whole plan.
