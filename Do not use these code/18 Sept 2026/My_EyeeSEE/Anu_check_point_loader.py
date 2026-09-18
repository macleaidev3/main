"""
======================================================================
EYEeSEE - STRICT BINARY LOCAL FUNDUS INFERENCE
======================================================================
Outputs strictly: "Glaucoma" OR "No Glaucoma"
======================================================================
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from model import UNet
from clinical_metrics_v2 import run_clinical_pipeline


# ======================================================================
#                     MANUAL PATH SETTINGS
# ======================================================================

FUNDUS_IMAGE_PATH = r"D:\Anurag BPCL WORK\Glaucosense\TASK NO. PRJ-002111\EyeeSEE\ESRGAN_Model_Result\1_OD_1_green.tif"
CHECKPOINT_PATH = r"D:\Anurag BPCL WORK\Glaucosense\TASK NO. PRJ-002111\EyeeSEE\My_EyeeSEE\checkpoints\checkpoint_epoch_057.pt"

# ======================================================================
# INFERENCE & CLINICAL THRESHOLDS
# ======================================================================

INPUT_SIZE = (512, 512)
MC_PASSES = 20
UNCERTAINTY_THRESHOLD = 0.05

DISC_THRESHOLD = 0.35
CUP_THRESHOLD = 0.35

# Strict binary clinical threshold (Standard clinical vCDR cutoff)
VCDR_GLAUCOMA_THRESHOLD = 0.50

# Maximum allowed cup pixel clipping before marking mask as invalid
MAX_ALLOWED_CUP_CLIPPING_RATIO = 0.40


# ======================================================================
# TERMINAL DISPLAY HELPERS
# ======================================================================

def separator(char="=", width=78):
    print(char * width)


def section(title):
    print()
    separator("-")
    print(title)
    separator("-")


# ======================================================================
# CHECKPOINT HANDLING
# ======================================================================

def resolve_checkpoint(model_path: str) -> Path:
    path = Path(model_path).expanduser().resolve()

    if path.is_file():
        if path.suffix.lower() not in [".pt", ".pth"]:
            raise ValueError(f"Checkpoint must be .pt or .pth:\n{path}")
        return path

    if not path.exists():
        raise FileNotFoundError(f"Model path does not exist:\n{path}")

    candidates = []
    candidates.extend(path.glob("checkpoint_epoch_*.pt"))
    candidates.extend(path.glob("checkpoint_epoch_*.pth"))

    if not candidates:
        candidates.extend(path.glob("*.pt"))
        candidates.extend(path.glob("*.pth"))

    if not candidates:
        raise FileNotFoundError(f"No PyTorch checkpoint found in:\n{path}")

    def epoch_number(p):
        match = re.search(r"checkpoint_epoch_(\d+)", p.name, re.IGNORECASE)
        return int(match.group(1)) if match else -1

    candidates.sort(key=lambda p: (epoch_number(p), p.name))
    return candidates[-1]


def load_checkpoint(model, checkpoint_path, device):
    print("\nLoading checkpoint...")
    print(f"Checkpoint path: {checkpoint_path}")

    checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)

    if not isinstance(checkpoint, dict):
        raise RuntimeError("Checkpoint is not a PyTorch dictionary.")

    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif checkpoint and all(torch.is_tensor(v) for v in checkpoint.values()):
        state_dict = checkpoint
    else:
        raise RuntimeError("Checkpoint does not contain 'model_state_dict' and is not a valid raw state_dict.")

    incompatible = model.load_state_dict(state_dict, strict=False)

    if incompatible.missing_keys:
        raise RuntimeError("Missing model parameters:\n" + "\n".join(incompatible.missing_keys))
    if incompatible.unexpected_keys:
        raise RuntimeError("Unexpected model parameters:\n" + "\n".join(incompatible.unexpected_keys))

    model.to(device)
    model.eval()
    print("Checkpoint loaded successfully.")
    return checkpoint


# ======================================================================
# PREPROCESSING & IMAGE READING
# ======================================================================

def read_original_image(image_path):
    path = Path(image_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Image does not exist:\n{path}")

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is not None:
        return image

    try:
        from PIL import Image
        with Image.open(path) as img:
            return np.array(img)
    except Exception as exc:
        raise RuntimeError(f"Unable to read image:\n{path}\n{exc}")


def convert_to_gray(image):
    if image.ndim == 2:
        return image, "GRAYSCALE"
    if image.ndim != 3:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    channels = image.shape[2]
    if channels == 1:
        return image[:, :, 0], "GRAYSCALE"
    if channels == 3:
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), "3-CHANNEL COLOR"
    if channels == 4:
        return cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2GRAY), "4-CHANNEL COLOR + ALPHA"

    raise ValueError(f"Unsupported number of channels: {channels}")


def normalize_real_pixels(gray):
    if gray.size == 0:
        raise ValueError("Image contains zero pixels.")

    if np.issubdtype(gray.dtype, np.integer):
        info = np.iinfo(gray.dtype)
        output = gray.astype(np.float32) / (255.0 if info.max <= 255 else float(info.max))
        return np.clip(output, 0.0, 1.0)

    if np.issubdtype(gray.dtype, np.floating):
        output = gray.astype(np.float32)
        if not np.isfinite(output).all():
            raise ValueError("Image contains NaN or Inf pixels.")
        minimum, maximum = float(output.min()), float(output.max())
        if minimum >= 0.0 and maximum <= 1.0:
            return output
        if minimum >= 0.0 and maximum <= 255.0:
            return output / 255.0

        raise ValueError(f"Unsupported floating-point intensity range: {minimum} .. {maximum}")

    raise TypeError(f"Unsupported image dtype: {gray.dtype}")


def prepare_image(image_path):
    original = read_original_image(image_path)
    original_shape = original.shape

    gray, image_type = convert_to_gray(original)
    gray01 = normalize_real_pixels(gray)

    resized = cv2.resize(gray01, INPUT_SIZE, interpolation=cv2.INTER_AREA)

    tensor = torch.from_numpy(resized.astype(np.float32)).unsqueeze(0).unsqueeze(0)

    metadata = {
        "original_shape": original_shape,
        "original_height": gray.shape[0],
        "original_width": gray.shape[1],
        "image_type": image_type,
        "dtype": str(original.dtype),
        "gray_min": float(gray01.min()),
        "gray_max": float(gray01.max()),
        "gray_mean": float(gray01.mean()),
        "gray_std": float(gray01.std()),
        "model_shape": tuple(tensor.shape),
    }

    return tensor, metadata


# ======================================================================
# MASK POST-PROCESSING & CLEANUP
# ======================================================================

def clean_mask(mask, min_size=3000):
    mask = mask.astype(np.uint8)
    if mask.sum() == 0:
        return mask

    kernel_size = 3 if mask.sum() < min_size else 5
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if count <= 1:
        return mask

    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    output = np.zeros_like(mask)
    output[labels == largest] = 1
    return output


def enforce_cup_inside_disc(disc_mask, cup_mask):
    corrected = np.logical_and(cup_mask == 1, disc_mask == 1).astype(np.uint8)
    violations = int(cup_mask.sum()) - int(corrected.sum())
    return disc_mask, corrected, violations


# ======================================================================
# MONTE-CARLO DROPOUT SEGMENTATION
# ======================================================================

def run_mc_segmentation(model, tensor, device, mc_passes, disc_threshold, cup_threshold):
    tensor = tensor.to(device)
    model.eval()

    for module in model.modules():
        if isinstance(module, (torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d)):
            module.train()

    all_probabilities = []

    with torch.no_grad():
        for _ in range(mc_passes):
            logits = model(tensor)
            probabilities = F.softmax(logits, dim=1)
            all_probabilities.append(probabilities.cpu().numpy())

    model.eval()

    stacked = np.stack(all_probabilities, axis=0)
    mean_probs = stacked.mean(axis=0)[0]
    variance_probs = stacked.var(axis=0)[0]

    disc_raw = (mean_probs[1] > disc_threshold).astype(np.uint8)
    cup_raw = (mean_probs[2] > cup_threshold).astype(np.uint8)

    raw_cup_pixels = int(cup_raw.sum())

    disc_mask, cup_mask, violations = enforce_cup_inside_disc(disc_raw, cup_raw)
    disc_mask = clean_mask(disc_mask, min_size=3000)
    cup_mask = clean_mask(cup_mask, min_size=1000)

    disc_mask, cup_mask, extra_violations = enforce_cup_inside_disc(disc_mask, cup_mask)
    total_violations = violations + extra_violations

    roi = (disc_mask == 1) | (cup_mask == 1)
    uncertainty = float(variance_probs[1:, roi].mean()) if roi.any() else float(variance_probs[1:].mean())

    return {
        "mean_probs": mean_probs,
        "variance_probs": variance_probs,
        "disc_raw": disc_raw,
        "cup_raw": cup_raw,
        "raw_cup_pixels": raw_cup_pixels,
        "disc_mask": disc_mask,
        "cup_mask": cup_mask,
        "violations": total_violations,
        "uncertainty": uncertainty,
    }


# ======================================================================
# REPORTING & DIRECT BINARY CLASSIFICATION
# ======================================================================

def print_report(image_path, checkpoint_path, metadata, checkpoint, device, segmentation, clinical):
    separator()
    print("EYEeSEE - DIRECT BINARY FUNDUS INFERENCE")
    separator()

    # Section 1 & 2
    section("1. INPUT IMAGE")
    print(f"Image path           : {image_path}")
    print(f"Original resolution  : {metadata['original_width']} x {metadata['original_height']}")
    print(f"Detected image type  : {metadata['image_type']}")

    section("2. MODEL & SEGMENTATION")
    print(f"Checkpoint           : {checkpoint_path}")
    print(f"Final disc pixels    : {int(segmentation['disc_mask'].sum()):,}")
    print(f"Final cup pixels     : {int(segmentation['cup_mask'].sum()):,}")
    print(f"Cup pixels clipped   : {segmentation['violations']:,}")

    # Section 3: Clinical Geometry
    section("3. CLINICAL GEOMETRY")
    print(f"vCDR                 : {clinical.vcdr:.4f}")
    print(f"ISNT rule satisfied  : {'YES' if clinical.isnt.rule_satisfied else 'NO'}")

    # Section 4: STRICT BINARY CLASSIFICATION LOGIC
    section("4. FINAL BINARY DIAGNOSIS")

    raw_cup_count = segmentation["raw_cup_pixels"]
    clipped_count = segmentation["violations"]
    clipping_ratio = (clipped_count / raw_cup_count) if raw_cup_count > 0 else 0.0

    # Guardrail Check
    if clipping_ratio > MAX_ALLOWED_CUP_CLIPPING_RATIO:
        final_stage = "No Glaucoma"
        diagnosis_reason = f"Mask corruption guardrail triggered (Excessive clipping: {clipping_ratio:.1%}). Defaulting safely."
    elif clinical.vcdr >= VCDR_GLAUCOMA_THRESHOLD:
        final_stage = "Glaucoma"
        diagnosis_reason = f"Elevated vCDR ({clinical.vcdr:.4f} >= {VCDR_GLAUCOMA_THRESHOLD})"
    elif not clinical.isnt.rule_satisfied and clinical.vcdr >= 0.40:
        final_stage = "Glaucoma"
        diagnosis_reason = f"ISNT rule violated with borderline vCDR ({clinical.vcdr:.4f})"
    else:
        final_stage = "No Glaucoma"
        diagnosis_reason = f"Normal vCDR ({clinical.vcdr:.4f} < {VCDR_GLAUCOMA_THRESHOLD}) and intact geometry."

    print(f"FINAL STAGE         : {final_stage}")
    print(f"Classification Basis: {diagnosis_reason}")
    separator()


# ======================================================================
# MAIN EXECUTION
# ======================================================================

def main():
    separator()
    print("Starting EyeeSEE Binary Inference...")
    separator()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device selected: {device}")

    tensor, metadata = prepare_image(FUNDUS_IMAGE_PATH)
    checkpoint_path = resolve_checkpoint(CHECKPOINT_PATH)

    model = UNet(in_channels=1, n_classes=3, base_filters=64, dropout=0.1)
    checkpoint = load_checkpoint(model=model, checkpoint_path=checkpoint_path, device=device)

    segmentation = run_mc_segmentation(
        model=model,
        tensor=tensor,
        device=device,
        mc_passes=MC_PASSES,
        disc_threshold=DISC_THRESHOLD,
        cup_threshold=CUP_THRESHOLD,
    )

    clinical = run_clinical_pipeline(
        disc_mask=segmentation["disc_mask"],
        cup_mask=segmentation["cup_mask"],
        uncertainty=segmentation["uncertainty"],
        uncertainty_threshold=UNCERTAINTY_THRESHOLD,
    )

    print_report(
        image_path=FUNDUS_IMAGE_PATH,
        checkpoint_path=str(checkpoint_path),
        metadata=metadata,
        checkpoint=checkpoint,
        device=device,
        segmentation=segmentation,
        clinical=clinical,
    )


if __name__ == "__main__":
    main()