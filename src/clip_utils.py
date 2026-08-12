"""Shared embedding extraction for the pretrained CLIP-style baseline.

Defaults to BiomedCLIP (Microsoft, PubMedBERT text tower + ViT-B/16 image tower,
trained on biomedical image-text pairs from PubMed Central) — the "pretrained
medical vision-language encoder" item in the project plan's minimum model stack.
Any other open_clip hf-hub model id can be swapped in via --model-id.
"""
from pathlib import Path

import open_clip
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

DEFAULT_MODEL_ID = "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"


def load_clip(model_id=DEFAULT_MODEL_ID, device="cpu"):
    model, preprocess = open_clip.create_model_from_pretrained(model_id)
    tokenizer = open_clip.get_tokenizer(model_id)
    model = model.to(device).eval()
    return model, preprocess, tokenizer


@torch.no_grad()
def embed_images(model, preprocess, img_root, paths, device, batch_size=16):
    img_root = Path(img_root)
    embeds = []
    for i in tqdm(range(0, len(paths), batch_size), desc="embedding images"):
        batch_paths = paths[i:i + batch_size]
        imgs = torch.stack([preprocess(Image.open(img_root / p).convert("RGB")) for p in batch_paths])
        feats = model.encode_image(imgs.to(device))
        feats = feats / feats.norm(dim=-1, keepdim=True)
        embeds.append(feats.cpu())
    return torch.cat(embeds, dim=0)


@torch.no_grad()
def embed_texts(model, tokenizer, texts, device, batch_size=32, context_length=256):
    embeds = []
    for i in tqdm(range(0, len(texts), batch_size), desc="embedding texts"):
        batch = texts[i:i + batch_size]
        tokens = tokenizer(batch, context_length=context_length).to(device)
        feats = model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        embeds.append(feats.cpu())
    return torch.cat(embeds, dim=0)


def load_pairs(csv_path):
    df = pd.read_csv(csv_path)
    assert "path" in df.columns and "report_text" in df.columns, \
        "CSV must have 'path' and 'report_text' columns (see prepare_iuxray_csv.py)"
    return df
