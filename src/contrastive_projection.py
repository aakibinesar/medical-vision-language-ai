"""A genuine (non-label-confounded) multimodal fusion test: does training on
this dataset's actual image-report pairs improve retrieval beyond generic
BiomedCLIP pretraining?

Unlike fusion_baseline.py (classification on the derived `Abnormal` label,
confounded by label-text leakage — see MODEL_CARD.md), this never touches
that label at all. It only uses the natural image<->report pairing, so
there's no leakage path: a real improvement here would be real evidence of
multimodal value, not a label-leakage artifact.

Keeps BiomedCLIP's backbone entirely frozen (avoids the "compute overload"
risk the project plan flags) and trains only two small linear projection
heads (image_dim -> proj_dim, text_dim -> proj_dim) with a symmetric
InfoNCE contrastive loss on the training set's embeddings, full-batch per
epoch since the embeddings are tiny once extracted. Model selection by val
Recall@1. Final comparison: zero-shot (raw BiomedCLIP embeddings, identity
projection) vs. trained-projection Recall@1/5/10 on the same held-out test
embeddings.
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from clip_utils import DEFAULT_MODEL_ID, embed_images, embed_texts, load_clip, load_pairs
from retrieval_baseline import evaluate_retrieval


class ProjectionHeads(nn.Module):
    def __init__(self, embed_dim, proj_dim):
        super().__init__()
        self.image_proj = nn.Linear(embed_dim, proj_dim, bias=False)
        self.text_proj = nn.Linear(embed_dim, proj_dim, bias=False)
        self.logit_scale = nn.Parameter(torch.tensor(float(torch.log(torch.tensor(1 / 0.07)))))

    def forward(self, image_embeds, text_embeds):
        img = F.normalize(self.image_proj(image_embeds), dim=-1)
        txt = F.normalize(self.text_proj(text_embeds), dim=-1)
        return img, txt

    def project(self, image_embeds, text_embeds):
        with torch.no_grad():
            return self.forward(image_embeds, text_embeds)


def contrastive_loss(img, txt, logit_scale):
    logits = logit_scale.exp().clamp(max=100) * img @ txt.T
    labels = torch.arange(logits.shape[0], device=logits.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", required=True, help="CSV with path,report_text columns")
    ap.add_argument("--val-csv", required=True)
    ap.add_argument("--test-csv", required=True)
    ap.add_argument("--img-root", required=True)
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    ap.add_argument("--batch-size", type=int, default=16, help="Batch size for embedding extraction only")
    ap.add_argument("--proj-dim", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess, tokenizer = load_clip(args.model_id, device)

    embeds = {}
    for name, csv_path in [("train", args.train_csv), ("val", args.val_csv), ("test", args.test_csv)]:
        df = load_pairs(csv_path)
        img_emb = embed_images(model, preprocess, args.img_root, df["path"].tolist(), device, args.batch_size)
        txt_emb = embed_texts(model, tokenizer, df["report_text"].tolist(), device, args.batch_size)
        embeds[name] = {"image": img_emb.to(device), "text": txt_emb.to(device), "n": len(df)}
        print(f"{name}: {len(df)} pairs embedded")

    embed_dim = embeds["train"]["image"].shape[1]
    heads = ProjectionHeads(embed_dim, args.proj_dim).to(device)
    optimizer = torch.optim.Adam(heads.parameters(), lr=args.lr)

    best_val_recall1 = -1.0
    best_state = None
    history = []
    for epoch in range(1, args.epochs + 1):
        heads.train()
        optimizer.zero_grad()
        img, txt = heads(embeds["train"]["image"], embeds["train"]["text"])
        loss = contrastive_loss(img, txt, heads.logit_scale)
        loss.backward()
        optimizer.step()

        heads.eval()
        val_img, val_txt = heads.project(embeds["val"]["image"], embeds["val"]["text"])
        val_metrics = evaluate_retrieval(val_img.cpu(), val_txt.cpu())
        val_r1 = val_metrics["image_to_text"].get("recall@1", 0.0)
        history.append({"epoch": epoch, "train_loss": float(loss.item()), "val_recall@1": val_r1})
        if epoch % 10 == 0 or epoch == 1:
            print(f"epoch {epoch}/{args.epochs}  loss={loss.item():.4f}  val_recall@1={val_r1:.4f}")
        if val_r1 > best_val_recall1:
            best_val_recall1 = val_r1
            best_state = {k: v.detach().clone() for k, v in heads.state_dict().items()}

    heads.load_state_dict(best_state)
    heads.eval()

    # Zero-shot: raw (un-projected, already-normalized) embeddings on the test set.
    zero_shot = evaluate_retrieval(embeds["test"]["image"].cpu(), embeds["test"]["text"].cpu())
    # Trained projection: same test embeddings, passed through the learned heads.
    proj_img, proj_txt = heads.project(embeds["test"]["image"], embeds["test"]["text"])
    trained = evaluate_retrieval(proj_img.cpu(), proj_txt.cpu())

    result = {
        "n_train": embeds["train"]["n"], "n_val": embeds["val"]["n"], "n_test": embeds["test"]["n"],
        "proj_dim": args.proj_dim, "epochs": args.epochs, "best_val_recall@1": best_val_recall1,
        "zero_shot": zero_shot,
        "trained_projection": trained,
        "improved": bool(trained["image_to_text"]["recall@1"] > zero_shot["image_to_text"]["recall@1"]),
    }
    print("\n=== Zero-shot (frozen BiomedCLIP, no training) ===")
    print(json.dumps(zero_shot, indent=2))
    print("\n=== Trained projection (contrastive fine-tuning on train pairs) ===")
    print(json.dumps(trained, indent=2))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "contrastive_projection.json", "w") as f:
        json.dump(result, f, indent=2)
    with open(outdir / "contrastive_projection_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"\nWrote {outdir / 'contrastive_projection.json'}")


if __name__ == "__main__":
    main()
