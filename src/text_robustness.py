"""Missing/noisy-text robustness test (mandatory evaluation block in the project
plan): corrupt or remove report text and measure how much retrieval performance
degrades relative to the clean baseline in retrieval_baseline.py.

If recall barely drops under corruption, that's a red flag — it would suggest the
model isn't actually using the text content. If recall collapses to near-chance
under "empty"/"mismatched", that's the expected, healthy behaviour: it confirms
retrieval genuinely depends on the report text rather than some other shortcut.
"""
import argparse
import json
import random
from pathlib import Path

import torch

from clip_utils import DEFAULT_MODEL_ID, embed_images, embed_texts, load_clip, load_pairs
from retrieval_baseline import evaluate_retrieval

CORRUPTIONS = ["original", "truncated_5_words", "shuffled_words", "empty", "mismatched"]


def corrupt_texts(texts, mode, seed=42):
    rng = random.Random(seed)
    if mode == "original":
        return list(texts)
    if mode == "truncated_5_words":
        return [" ".join(t.split()[:5]) for t in texts]
    if mode == "shuffled_words":
        out = []
        for t in texts:
            words = t.split()
            rng.shuffle(words)
            out.append(" ".join(words))
        return out
    if mode == "empty":
        return [""] * len(texts)
    if mode == "mismatched":
        shuffled = list(texts)
        rng.shuffle(shuffled)
        return shuffled
    raise ValueError(f"Unknown corruption mode: {mode}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV with path,report_text columns")
    ap.add_argument("--img-root", required=True)
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--max-examples", type=int, default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = load_pairs(args.csv)
    if args.max_examples:
        df = df.sample(n=min(args.max_examples, len(df)), random_state=42).reset_index(drop=True)

    model, preprocess, tokenizer = load_clip(args.model_id, device)
    # Image embeddings don't change across corruptions — compute once.
    image_embeds = embed_images(model, preprocess, args.img_root, df["path"].tolist(), device, args.batch_size)

    results = {}
    for mode in CORRUPTIONS:
        texts = corrupt_texts(df["report_text"].tolist(), mode, args.seed)
        text_embeds = embed_texts(model, tokenizer, texts, device, args.batch_size)
        results[mode] = evaluate_retrieval(image_embeds, text_embeds)
        r1 = results[mode]["image_to_text"].get("recall@1", float("nan"))
        print(f"{mode:20s}  image->text recall@1 = {r1:.3f}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "text_robustness.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {outdir / 'text_robustness.json'}")


if __name__ == "__main__":
    main()
