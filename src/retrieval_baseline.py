"""Zero-shot image<->report retrieval using a pretrained CLIP-style model
(BiomedCLIP by default). This is the "pretrained medical vision-language
encoder / CLIP-style model" item in the minimum model stack, and the
"image-report retrieval" task family the project plan recommends as a
starting multimodal task (cleaner than deriving noisy classification labels
from report text).

No training happens here — this measures how well an off-the-shelf medical
VLM already aligns images and reports on this dataset, which is the right
first multimodal data point before spending compute on fine-tuning.
"""
import argparse
import json
from pathlib import Path

import torch

from clip_utils import DEFAULT_MODEL_ID, embed_images, embed_texts, load_clip, load_pairs


def recall_at_k(sim, ks=(1, 5, 10)):
    """sim[i, j] = similarity of query i to candidate j; correct match is the diagonal."""
    n = sim.shape[0]
    ranks = sim.argsort(dim=1, descending=True)
    correct_rank = (ranks == torch.arange(n).unsqueeze(1)).nonzero()[:, 1]
    return {f"recall@{k}": float((correct_rank < k).float().mean()) for k in ks if k <= n}


def evaluate_retrieval(image_embeds, text_embeds):
    sim = image_embeds @ text_embeds.T
    return {
        "image_to_text": recall_at_k(sim),
        "text_to_image": recall_at_k(sim.T),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV with path,report_text columns")
    ap.add_argument("--img-root", required=True)
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-examples", type=int, default=None,
                     help="Subsample for a quick run (retrieval cost grows with N)")
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = load_pairs(args.csv)
    if args.max_examples:
        df = df.sample(n=min(args.max_examples, len(df)), random_state=42).reset_index(drop=True)

    model, preprocess, tokenizer = load_clip(args.model_id, device)
    image_embeds = embed_images(model, preprocess, args.img_root, df["path"].tolist(), device, args.batch_size)
    text_embeds = embed_texts(model, tokenizer, df["report_text"].tolist(), device, args.batch_size)

    metrics = evaluate_retrieval(image_embeds, text_embeds)
    metrics["n_examples"] = len(df)
    metrics["model_id"] = args.model_id
    print(json.dumps(metrics, indent=2))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "retrieval_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"Wrote {outdir / 'retrieval_metrics.json'}")


if __name__ == "__main__":
    main()
