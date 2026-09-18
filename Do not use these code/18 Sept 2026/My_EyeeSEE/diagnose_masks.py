"""
Drop this file next to phase3pipeline.py and run it once on any fundus image.
It prints the raw probability statistics and saves visual debug images
so you can see exactly what the model is predicting before any thresholding.

Usage:
    python diagnose_masks.py path/to/fundus.jpg
"""

import sys
import os
import numpy as np
import cv2
import torch
import torch.nn.functional as F

# ── adjust these imports to match your project layout ──────────────────
from model import UNet
from checkpoint_loader import load_model_for_inference
# ───────────────────────────────────────────────────────────────────────

REPO_ID = "Nj-1111/EyeeSEE"
TOKEN   = os.getenv("HF_TOKEN_2") or os.getenv("HF_TOKEN")
DEVICE  = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model():
    model = UNet(in_channels=1, n_classes=3, base_filters=64, dropout=0.1)
    load_model_for_inference(
        model=model, repo_id=REPO_ID, epoch=None,
        device=DEVICE, token=TOKEN
    )
    model.eval()
    return model


def preprocess(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (512, 512), interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(gray.astype(np.float32) / 255.0)
    return t.unsqueeze(0).unsqueeze(0).to(DEVICE)


def run(image_path):
    model  = load_model()
    tensor = preprocess(image_path)

    # single deterministic pass (no MC dropout) — we want the raw probs
    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1).cpu().numpy()[0]  # (3, H, W)

    bg   = probs[0]   # background
    disc = probs[1]   # optic disc
    cup  = probs[2]   # optic cup

    print("=" * 56)
    print("  RAW PROBABILITY DIAGNOSTICS")
    print("=" * 56)

    for name, ch in [("background", bg), ("disc     ", disc), ("cup      ", cup)]:
        print(
            f"  {name}  "
            f"min={ch.min():.4f}  "
            f"max={ch.max():.4f}  "
            f"mean={ch.mean():.4f}  "
            f"median={np.median(ch):.4f}"
        )

    print()

    # How many pixels survive each threshold?
    print("  Disc pixels above threshold:")
    for t in [0.10, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50]:
        print(f"    > {t:.2f}  →  {(disc > t).sum():6d} px")

    print()
    print("  Cup pixels above threshold:")
    for t in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.55, 0.60]:
        print(f"    > {t:.2f}  →  {(cup > t).sum():6d} px")

    print()

    # Spatial overlap — at each threshold pair, how many cup pixels are
    # INSIDE the disc mask?  This tells you whether the predictions are
    # spatially aligned.
    print("  Spatial alignment (cup pixels inside disc) at disc>0.35:")
    disc_bin = (disc > 0.35).astype(np.uint8)
    for t in [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.55]:
        cup_bin  = (cup > t).astype(np.uint8)
        inside   = int((cup_bin & disc_bin).sum())
        total    = int(cup_bin.sum())
        pct      = (inside / total * 100) if total > 0 else 0.0
        print(
            f"    cup>{t:.2f}  total={total:5d}  "
            f"inside_disc={inside:5d}  ({pct:.1f}%)"
        )

    print("=" * 56)

    # Save visual probability maps (scaled 0-255)
    cv2.imwrite("debug_bg_prob.png",   (bg   * 255).astype(np.uint8))
    cv2.imwrite("debug_disc_prob.png", (disc * 255).astype(np.uint8))
    cv2.imwrite("debug_cup_prob.png",  (cup  * 255).astype(np.uint8))

    # Save side-by-side composite
    disc_vis = cv2.applyColorMap((disc * 255).astype(np.uint8), cv2.COLORMAP_HOT)
    cup_vis  = cv2.applyColorMap((cup  * 255).astype(np.uint8), cv2.COLORMAP_COOL)
    composite = np.hstack([disc_vis, cup_vis])
    cv2.imwrite("debug_composite.png", composite)

    print()
    print("  Saved: debug_disc_prob.png  debug_cup_prob.png  debug_composite.png")
    print("  Open these images to see where the model is predicting disc and cup.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python diagnose_masks.py <path_to_fundus_image>")
        sys.exit(1)
    run(sys.argv[1])
