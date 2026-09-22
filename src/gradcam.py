"""Minimal Grad-CAM (Selvaraju et al. 2017) for the resnet18/densenet121
baselines here — hooks the last convolutional block, no extra dependency.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dataset import build_transforms
from train import build_model


def get_last_conv_layer(model, model_name):
    if model_name == "resnet18":
        return model.layer4[-1]
    if model_name == "densenet121":
        return model.features.norm5
    raise ValueError(f"Unknown model: {model_name}")


class GradCAM:
    """Hooks the target layer's output tensor directly (via `Tensor.register_hook`)
    rather than `Module.register_full_backward_hook`. DenseNet121 applies an
    in-place ReLU immediately after `features.norm5`, and a full backward hook
    wraps the module's output in a way that conflicts with that later in-place
    op ("view + inplace" autograd error) — a tensor-level hook avoids the
    wrapping entirely and works for any architecture.
    """

    def __init__(self, model, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None
        target_layer.register_forward_hook(self._save_activation)

    def _save_activation(self, module, inp, out):
        self.activations = out
        out.register_hook(self._save_gradient)

    def _save_gradient(self, grad):
        self.gradients = grad.detach()

    def __call__(self, x, class_idx=0):
        self.model.zero_grad()
        logits = self.model(x)
        score = logits[:, class_idx].sum()
        score.backward()
        activations = self.activations.detach()
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * activations).sum(dim=1))
        cam = cam - cam.amin(dim=(1, 2), keepdim=True)
        cam = cam / (cam.amax(dim=(1, 2), keepdim=True) + 1e-8)
        return cam  # (B, H, W) in [0, 1]


def overlay(pil_img, cam, alpha=0.4):
    cam_resized = np.array(
        Image.fromarray((cam * 255).astype("uint8")).resize(pil_img.size, Image.BILINEAR)
    ) / 255.0
    heatmap = plt.get_cmap("jet")(cam_resized)[:, :, :3]
    base = np.array(pil_img.convert("RGB")) / 255.0
    return np.clip((1 - alpha) * base + alpha * heatmap, 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--img-root", required=True)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n-examples", type=int, default=6)
    ap.add_argument("--include-path", nargs="*", default=None,
                     help="Force these specific `path` values into the sample (e.g. to "
                          "re-render a known example under a different checkpoint), on top "
                          "of the usual random sample.")
    ap.add_argument("--out", default="results/gradcam_examples")
    args = ap.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model_name = ckpt["model_name"]
    img_size = ckpt["img_size"]
    label_name = ckpt["label_columns"][0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(model_name, len(ckpt["label_columns"]),
                         dropout_p=ckpt.get("dropout_p", 0.0)).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    cam_engine = GradCAM(model, get_last_conv_layer(model, model_name))
    tf = build_transforms(img_size, train=False)

    full_df = pd.read_csv(args.csv)
    df = full_df.sample(n=min(args.n_examples, len(full_df)), random_state=42)
    if args.include_path:
        forced = full_df[full_df["path"].isin(args.include_path)]
        df = pd.concat([df, forced]).drop_duplicates(subset="path")
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    img_root = Path(args.img_root)

    for _, row in df.iterrows():
        pil_img = Image.open(img_root / row["path"]).convert("RGB")
        x = tf(pil_img).unsqueeze(0).to(device)
        with torch.enable_grad():
            probs = torch.sigmoid(model(x)).detach().cpu().numpy()[0]
            cam = cam_engine(x, class_idx=0)[0].cpu().numpy()
        overlay_img = overlay(pil_img, cam)

        true_label = int(row.get(label_name, -1))
        pred_prob = probs[0]
        fname = Path(row["path"]).stem
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.imshow(overlay_img)
        ax.axis("off")
        ax.set_title(f"true={true_label}  p({label_name.lower()})={pred_prob:.2f}", fontsize=9)
        fig.tight_layout()
        fig.savefig(out_dir / f"{fname}_cam.png", dpi=150)
        plt.close(fig)

    print(f"Wrote {len(df)} Grad-CAM overlays to {out_dir}/")


if __name__ == "__main__":
    main()
