"""Extends contrastive_projection.py's genuine (non-label-confounded) fusion
test by additionally fine-tuning the last N transformer blocks of BiomedCLIP's
image and text towers, instead of keeping the whole backbone frozen.

contrastive_projection.py already answers "does *any* training on this
dataset's real pairs beat zero-shot BiomedCLIP" (yes - roughly doubles
Recall@5/@10, confirmed across seeds). Its own limitations section leaves a
follow-up question open: does letting the backbone itself adapt - not just
two linear heads on top of a frozen backbone - help further, or does
adapting several million extra parameters over only ~2,568 real training
pairs overfit instead? This script is that follow-up experiment, not a
replacement for the frozen one - both results should be reported side by
side.

Differences from contrastive_projection.py, and why:
- Can't reuse the "extract embeddings once, reuse across seeds" trick, since
  the backbone itself is trainable here - embeddings change every step.
  Trains in mini-batches over raw images/text instead of one full-batch
  step per epoch over cached embeddings.
- The backbone gets its own, much smaller learning rate than the projection
  heads (--backbone-lr, default 1e-6, vs --head-lr default 1e-3) - standard
  practice when fine-tuning a large pretrained network on a small amount of
  new data, to avoid destroying pretrained representations in a few noisy
  steps.
- The whole model is kept in eval() mode throughout, even during training -
  deliberate, not an oversight: with only ~2,568 training pairs, adding
  dropout noise on top of an already overfitting-prone setup seemed like
  the wrong default. Report this alongside the results, don't hide it.
- Each seed resets the backbone to its original pretrained weights before
  fine-tuning again (otherwise seed 2 would fine-tune on top of seed 1's
  already-adapted weights, contaminating the seed-to-seed comparison this
  project relies on everywhere else).
"""
import argparse
import json
import time
from pathlib import Path

import torch

from clip_utils import (
    DEFAULT_MODEL_ID, embed_images, embed_texts, encode_images_batch,
    encode_texts_batch, load_clip, load_pairs, unfreeze_last_layers,
)
from contrastive_projection import ProjectionHeads, aggregate_across_seeds, contrastive_loss
from retrieval_baseline import evaluate_retrieval


def run_one_seed(seed, model, orig_state, preprocess, tokenizer, dfs, img_root, device,
                  proj_dim, embed_dim, epochs, head_lr, backbone_lr, batch_size,
                  eval_batch_size, unfreeze_layers, verbose):
    torch.manual_seed(seed)
    model.load_state_dict(orig_state)
    unfreeze_last_layers(model, unfreeze_layers)
    heads = ProjectionHeads(embed_dim, proj_dim).to(device)

    backbone_params = [p for p in model.parameters() if p.requires_grad]
    trainable_keys = {k for k, p in model.named_parameters() if p.requires_grad}
    optimizer = torch.optim.Adam([
        {"params": backbone_params, "lr": backbone_lr},
        {"params": heads.parameters(), "lr": head_lr},
    ])

    train_df = dfs["train"]
    n = len(train_df)
    best_val_recall1 = -1.0
    best_state = None
    history = []

    for epoch in range(1, epochs + 1):
        epoch_t0 = time.time()
        model.eval()  # dropout off everywhere; requires_grad alone controls what trains
        heads.train()
        perm = torch.randperm(n).tolist()
        total_loss, n_batches = 0.0, 0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            batch_paths = [train_df["path"].iloc[j] for j in idx]
            batch_texts = [train_df["report_text"].iloc[j] for j in idx]
            optimizer.zero_grad()
            img_embed = encode_images_batch(model, preprocess, img_root, batch_paths, device)
            txt_embed = encode_texts_batch(model, tokenizer, batch_texts, device)
            img, txt = heads(img_embed, txt_embed)
            loss = contrastive_loss(img, txt, heads.logit_scale)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        avg_loss = total_loss / max(n_batches, 1)

        model.eval()
        heads.eval()
        with torch.no_grad():
            val_img_raw = embed_images(model, preprocess, img_root, dfs["val"]["path"].tolist(),
                                        device, eval_batch_size)
            val_txt_raw = embed_texts(model, tokenizer, dfs["val"]["report_text"].tolist(),
                                       device, eval_batch_size)
            val_img, val_txt = heads.project(val_img_raw.to(device), val_txt_raw.to(device))
        val_metrics = evaluate_retrieval(val_img.cpu(), val_txt.cpu())
        val_r1 = val_metrics["image_to_text"].get("recall@1", 0.0)
        epoch_seconds = round(time.time() - epoch_t0, 1)
        history.append({"epoch": epoch, "train_loss": avg_loss, "val_recall@1": val_r1,
                         "seconds": epoch_seconds})
        if verbose:
            print(f"  [seed {seed}] epoch {epoch}/{epochs}  loss={avg_loss:.4f}  "
                  f"val_recall@1={val_r1:.4f}  ({epoch_seconds}s)")
        if val_r1 > best_val_recall1:
            best_val_recall1 = val_r1
            best_state = {
                "heads": {k: v.detach().clone() for k, v in heads.state_dict().items()},
                "backbone": {k: v.detach().clone() for k, v in model.state_dict().items()
                             if k in trainable_keys},
            }

    heads.load_state_dict(best_state["heads"])
    model.load_state_dict(best_state["backbone"], strict=False)
    model.eval()
    heads.eval()
    with torch.no_grad():
        test_img_raw = embed_images(model, preprocess, img_root, dfs["test"]["path"].tolist(),
                                     device, eval_batch_size)
        test_txt_raw = embed_texts(model, tokenizer, dfs["test"]["report_text"].tolist(),
                                    device, eval_batch_size)
        test_img, test_txt = heads.project(test_img_raw.to(device), test_txt_raw.to(device))
    trained = evaluate_retrieval(test_img.cpu(), test_txt.cpu())
    best_epoch = max(range(len(history)), key=lambda i: history[i]["val_recall@1"]) + 1
    return trained, best_val_recall1, best_epoch, history


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-csv", required=True, help="CSV with path,report_text columns")
    ap.add_argument("--val-csv", required=True)
    ap.add_argument("--test-csv", required=True)
    ap.add_argument("--img-root", required=True)
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    ap.add_argument("--proj-dim", type=int, default=256)
    ap.add_argument("--unfreeze-layers", type=int, default=2,
                     help="Number of final transformer blocks to unfreeze in EACH tower "
                          "(image ViT + text BERT, 12 blocks each). 0 = fully frozen - use "
                          "contrastive_projection.py directly for that, it's far cheaper.")
    ap.add_argument("--epochs", type=int, default=15,
                     help="Fewer than contrastive_projection.py's default (100-200): each "
                          "epoch here is a real mini-batch pass with backbone gradients, "
                          "not one cheap full-batch step over cached embeddings.")
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--backbone-lr", type=float, default=1e-6,
                     help="Deliberately much smaller than --head-lr - the backbone is "
                          "pretrained and should move slowly; the heads start from random "
                          "init and need to move fast.")
    ap.add_argument("--batch-size", type=int, default=32,
                     help="Mini-batch size for the fine-tuning forward/backward pass.")
    ap.add_argument("--eval-batch-size", type=int, default=16,
                     help="Batch size for the frozen-mode embedding extraction used for "
                          "zero-shot and for validation/test evaluation.")
    ap.add_argument("--seeds", type=int, nargs="+", default=[42])
    ap.add_argument("--outdir", default="results")
    args = ap.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, preprocess, tokenizer = load_clip(args.model_id, device)
    orig_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    dfs = {name: load_pairs(csv) for name, csv in
           [("train", args.train_csv), ("val", args.val_csv), ("test", args.test_csv)]}
    for name, df in dfs.items():
        print(f"{name}: {len(df)} pairs")

    with torch.no_grad():
        zs_img = embed_images(model, preprocess, args.img_root, dfs["test"]["path"].tolist(),
                               device, args.eval_batch_size)
        zs_txt = embed_texts(model, tokenizer, dfs["test"]["report_text"].tolist(),
                              device, args.eval_batch_size)
    zero_shot = evaluate_retrieval(zs_img.cpu(), zs_txt.cpu())
    embed_dim = zs_img.shape[1]
    print("\n=== Zero-shot (frozen BiomedCLIP, no training) ===")
    print(json.dumps(zero_shot, indent=2))

    per_seed_trained, per_seed_meta = [], []
    for seed in args.seeds:
        print(f"\n--- seed {seed} (unfreezing last {args.unfreeze_layers} block(s) per tower) ---")
        trained, best_val_r1, best_epoch, history = run_one_seed(
            seed, model, orig_state, preprocess, tokenizer, dfs, args.img_root, device,
            args.proj_dim, embed_dim, args.epochs, args.head_lr, args.backbone_lr,
            args.batch_size, args.eval_batch_size, args.unfreeze_layers, verbose=True)
        per_seed_trained.append(trained)
        per_seed_meta.append({"seed": seed, "best_val_recall@1": best_val_r1, "best_epoch": best_epoch})
        print(f"seed {seed} test result: {json.dumps(trained)}")
        if len(args.seeds) == 1:
            with open(Path(args.outdir) / "contrastive_finetune_history.json", "w") as f:
                json.dump(history, f, indent=2)

    trained_agg = aggregate_across_seeds(per_seed_trained)
    result = {
        "n_train": len(dfs["train"]), "n_val": len(dfs["val"]), "n_test": len(dfs["test"]),
        "proj_dim": args.proj_dim, "unfreeze_layers": args.unfreeze_layers,
        "epochs": args.epochs, "head_lr": args.head_lr, "backbone_lr": args.backbone_lr,
        "batch_size": args.batch_size, "seeds": args.seeds,
        "per_seed": per_seed_meta,
        "zero_shot": zero_shot,
        "finetuned": per_seed_trained[0] if len(args.seeds) == 1 else None,
        "finetuned_per_seed": per_seed_trained if len(args.seeds) > 1 else None,
        "finetuned_mean_std": trained_agg if len(args.seeds) > 1 else None,
    }
    print(f"\n=== Fine-tuned ({'mean +/- std over ' + str(len(args.seeds)) + ' seeds' if len(args.seeds) > 1 else 'single seed'}) ===")
    print(json.dumps(trained_agg if len(args.seeds) > 1 else per_seed_trained[0], indent=2))

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    with open(outdir / "contrastive_finetune.json", "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {outdir / 'contrastive_finetune.json'}")


if __name__ == "__main__":
    main()
