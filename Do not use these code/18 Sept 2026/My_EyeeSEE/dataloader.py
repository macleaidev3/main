# ── Cell 1: write dataloader.py ─────────────────────────────────────────────
import os
import cv2
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset


class GlaucomaDataset(Dataset):
    """
    Local-disk fundus dataset with image-driven synthetic mask generation.

    Mask convention
    ---------------
    0 = background  |  1 = optic disc  |  2 = optic cup

    ONH centre estimated per-image from the brightest region of the image,
    so every sample gets a unique mask.  Training split applies augmentation.
    """

    def __init__(self, root_dir: str, split: str = "train"):
        self.root_dir = Path(root_dir)
        self.split    = split
        self.augment  = (split == "train")
        self.image_paths: list = []
        self.labels:      list = []
        self._load_index()

    def _split_folder(self) -> str:
        return {"train": "train set", "validation": "validation set",
                "test": "test set"}.get(self.split, "train set")

    def _load_index(self):
        base = self.root_dir / self._split_folder()
        for label, sub in [(0, "nrg"), (1, "rg")]:
            folder = base / sub
            if folder.exists():
                for p in folder.rglob("*"):
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png"):
                        self.image_paths.append(str(p))
                        self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    # ── ONH detection ─────────────────────────────────────────────────────────

    def _onh_center(self, gray_u8: np.ndarray):
        h, w = gray_u8.shape
        my, mx = int(h * 0.15), int(w * 0.15)
        roi = gray_u8[my: h - my, mx: w - mx]

        # Large blur suppresses point reflections — a real ONH is a region, not a point
        blurred = cv2.GaussianBlur(roi, (61, 61), 0)

        # Centroid of top-5% brightest region is robust to specular artifacts
        threshold = np.percentile(blurred, 95)
        bright = (blurred >= threshold).astype(np.uint8)

        n, labels, stats, centroids = cv2.connectedComponentsWithStats(bright)
        if n > 1:
            largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            cx = int(centroids[largest][0]) + mx
            cy = int(centroids[largest][1]) + my
        else:
            cy, cx = h // 2, w // 2

        return cy, cx

    
    def _make_mask(self, gray_f32: np.ndarray) -> np.ndarray:
        h, w   = gray_f32.shape
        cy, cx = self._onh_center((gray_f32 * 255).astype(np.uint8))

        base_r = int(min(h, w) * 0.13)

        if self.augment:
            r_jitter = int(base_r * 0.12)
            disc_r   = base_r + np.random.randint(-r_jitter, r_jitter + 1)
            cup_r    = int(disc_r * np.random.uniform(0.60, 0.72))
            cj       = int(base_r * 0.04)
            cx       = int(np.clip(cx + np.random.randint(-cj, cj + 1), disc_r, w - disc_r))
            cy       = int(np.clip(cy + np.random.randint(-cj, cj + 1), disc_r, h - disc_r))
        else:
            disc_r = base_r
            cup_r  = int(disc_r * 0.55)
            cx     = int(np.clip(cx, disc_r, w - disc_r))
            cy     = int(np.clip(cy, disc_r, h - disc_r))



        y, x = np.ogrid[:h, :w]
        d    = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[d <= disc_r] = 1
        mask[d <= cup_r]  = 2
        return mask

    # ── augmentation ──────────────────────────────────────────────────────────

    def _augment(self, img: np.ndarray, mask: np.ndarray):
        if np.random.random() > 0.5:
            img, mask = cv2.flip(img, 1), cv2.flip(mask, 1)
        if np.random.random() > 0.5:
            img, mask = cv2.flip(img, 0), cv2.flip(mask, 0)

        angle = np.random.uniform(-7, 7)
        h, w  = img.shape
        M     = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        kw_i  = dict(flags=cv2.INTER_LINEAR,   borderMode=cv2.BORDER_REFLECT_101)
        kw_m  = dict(flags=cv2.INTER_NEAREST,  borderMode=cv2.BORDER_REFLECT_101)
        img   = cv2.warpAffine(img,  M, (w, h), **kw_i)
        mask  = cv2.warpAffine(mask, M, (w, h), **kw_m)

        alpha = np.random.uniform(0.92, 1.08)
        beta  = np.random.uniform(-0.02, 0.02)
        img   = np.clip(img * alpha + beta, 0.0, 1.0).astype(np.float32)

        if np.random.random() > 0.5:
            img = np.clip(img + np.random.normal(0, 0.005, img.shape).astype(np.float32),
                          0.0, 1.0)
        return img, mask

    # ── __getitem__ ────────────────────────────────────────────────────────────

    def __getitem__(self, idx: int):
        img = cv2.imread(self.image_paths[idx], cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((512, 512), dtype=np.uint8)
        img  = cv2.resize(img, (512, 512)).astype(np.float32) / 255.0
        mask = self._make_mask(img)
        if self.augment:
            img, mask = self._augment(img, mask)
        return (torch.from_numpy(img).unsqueeze(0).float(),
                torch.from_numpy(mask).long())
