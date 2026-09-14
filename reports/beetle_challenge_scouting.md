# BEETLE Challenge — Scouting Report

**Status: scouting only, no code/submission started.** Checked directly against
[beetle.grand-challenge.org](https://beetle.grand-challenge.org/), its
[GitHub repo](https://github.com/DIAGNijmegen/beetle), and the
[Zenodo record](https://zenodo.org/records/16812932) on 2026-09-03.

## Task

4-class semantic segmentation of H&E-stained breast cancer whole-slide images
(WSIs): invasive epithelium, non-invasive epithelium, necrosis, other.
Development set: 587 biopsies/resections, 3 clinical centers + 2 public
datasets, 7 scanner types. Evaluation set: 170 densely-annotated ROIs from 54
WSIs, 3 scanners. No deadline — explicitly a "long-term benchmark," not a
dated competition.

## The blocker: dataset size

The Zenodo record is **150.9GB total**: `images.zip` alone is **147.2GB**,
plus `annotations.zip` (1.8GB) and `model.zip` (1.9GB, a pretrained baseline
ensemble). I could not confirm, after checking the challenge site, the
GitHub repo, and the Zenodo file listing directly, **whether the 170
evaluation ROIs can be downloaded separately from the 147GB archive** — the
challenge page says "download the 170 ROI images... from Zenodo" as if it's
a standalone action, but the Zenodo record itself only lists one combined
`images.zip`. This is the single fact that determines whether a first
submission is feasible at all on typical local/free-tier cloud storage — it
needs a direct question to Zenodo support or the challenge organizers
(contact via the GitHub repo issues) before spending any more time on this.

For scale: your local disk has ~50GB free. Kaggle Notebooks and free Colab
both cap working-directory disk well under 147GB too. This would need either
confirmation that a small eval-only download exists, or a paid cloud VM with
more disk — not a "just run it" afternoon task as it first appeared.

## License

**CC BY-NC-SA 4.0** (NonCommercial-ShareAlike) — fine for a non-commercial
PhD portfolio, but ShareAlike means anything built on this data and
redistributed (e.g. a fine-tuned model checkpoint) would need to carry the
same license. Not a blocker, just a term to respect if publishing derived
models.

## Submission mechanism

Standard grand-challenge.org pattern, not a simple file upload: package your
inference code as a **Docker container** (the GitHub repo ships a
`code/docker/Dockerfile` and `run_inference.sh` as a template), register it
as an "Algorithm" on the platform, then submit that algorithm to run against
BEETLE's hidden test set. Requires a free grand-challenge.org account
(registration page is login-gated — you'd need to create this yourself).
This container-submission step is itself real engineering work, separate
from the modeling.

## What's actually promising here

- A **pretrained baseline ensemble model is provided** (`model.zip`) —
  reported overall Dice 0.87 (per-class: other 0.94, invasive epithelium
  0.78, non-invasive epithelium 0.65, necrosis 0.51). In principle, if the
  eval-only download question resolves favorably, a first "valid submission"
  could be running their baseline through the Docker pipeline unmodified —
  exactly the kind of low-effort first submission the two-project red-team
  plan recommends before improving anything.
- No deadline pressure, unlike CLEAR-EC/REG2026.
- Genuinely relevant to Slabaugh's feedback (computer vision, real benchmark
  participation) and to Project 2's pathology domain, more than to Project
  1's current VLM/retrieval focus.

## Recommendation

**Don't commit to this yet.** The 147GB question is a hard blocker until
resolved — email/file a GitHub issue asking the DIAGNijmegen team whether the
170 ROI evaluation images are downloadable independent of the full
development archive. If yes: this becomes a cheap, no-deadline, baseline-
provided entry point. If no: this challenge needs cloud storage/compute this
setup doesn't currently have, and is not the right near-term target —
REG2027 (once it opens) or a lighter-weight challenge would be a better use
of the "benchmark participation" line item from Slabaugh's feedback.
