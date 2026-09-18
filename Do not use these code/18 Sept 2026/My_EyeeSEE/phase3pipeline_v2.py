"""
Phase 3 — Inference Pipeline
"""

import os
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple

from model import UNet
from checkpoint_loader import load_model_for_inference
from clinical_metrics_v2 import run_clinical_pipeline, ClinicalResult

_MIN_DISC_AREA_PX   = 2_600
_MIN_CUP_AREA_PX    = 100
_MIN_CUP_DISC_RATIO = 0.01


class Phase3Pipeline_v2:

    def __init__(
        self,
        repo_id: str = "Nj-1111/EyeeSEE",
        epoch: Optional[int] = None,
        mc_passes: int = 20,
        uncertainty_threshold: float = 0.05,
        device: Optional[torch.device] = None,
        token: Optional[str] = None,
        debug: bool = False,
        disc_threshold: float = 0.35,
        cup_threshold:  float = 0.35,   # start here; tune down if cup stays 0
    ):
        self.repo_id               = repo_id
        self.mc_passes             = mc_passes
        self.uncertainty_threshold = uncertainty_threshold
        self.debug                 = debug
        self.disc_threshold        = disc_threshold
        self.cup_threshold         = cup_threshold

        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.token = (
            token or
            os.getenv('HF_TOKEN_2') or
            os.getenv('HF_TOKEN')
        )

        self.model = UNet(
            in_channels=1,
            n_classes=3,
            base_filters=64,
            dropout=0.1
        )
        load_model_for_inference(
            model=self.model,
            repo_id=repo_id,
            epoch=epoch,
            device=self.device,
            token=self.token
        )

    # ──────────────────────────────────────────────────────────────────
    # preprocessing
    # ──────────────────────────────────────────────────────────────────

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:
        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)
        image = image.astype(np.float32) / 255.0
        return (
            torch.from_numpy(image)
            .unsqueeze(0).unsqueeze(0)
            .to(self.device)
        )

    # ──────────────────────────────────────────────────────────────────
    # mask cleanup
    # ──────────────────────────────────────────────────────────────────

    def _clean_disc(self, mask: np.ndarray) -> np.ndarray:
        mask = mask.astype(np.uint8)
        if mask.sum() == 0:
            return mask
        k = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        if n <= 1:
            return mask
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        out = np.zeros_like(mask)
        out[labels == largest] = 1
        return out

    def _clean_cup(self, mask: np.ndarray) -> np.ndarray:
        mask = mask.astype(np.uint8)
        if mask.sum() == 0:
            return mask
        # Adaptive kernel — a large kernel erases small post-clip remnants
        k_size = 3 if mask.sum() < 3_000 else 5
        k = np.ones((k_size, k_size), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
        if n <= 1:
            return mask
        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        out = np.zeros_like(mask)
        out[labels == largest] = 1
        return out

    # ──────────────────────────────────────────────────────────────────
    # anatomical enforcement
    # ──────────────────────────────────────────────────────────────────

    def _enforce_cup_in_disc(
        self,
        disc_mask: np.ndarray,
        cup_mask:  np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, int]:
        corrected = np.logical_and(cup_mask == 1, disc_mask == 1).astype(np.uint8)
        violations = int(cup_mask.sum()) - int(corrected.sum())
        return disc_mask, corrected, violations

    # ──────────────────────────────────────────────────────────────────
    # MC-Dropout segmentation  — returns RAW unprocessed masks + var map
    # ──────────────────────────────────────────────────────────────────

    def _mc_segment(
        self,
        tensor: torch.Tensor,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns:
            disc_mask  — thresholded, NOT yet cleaned
            cup_mask   — thresholded, NOT yet cleaned
            var_probs  — (3, H, W) full variance map for ROI uncertainty
        """
        self.model.eval()
        for m in self.model.modules():
            if isinstance(m, (torch.nn.Dropout, torch.nn.Dropout2d, torch.nn.Dropout3d)):
                m.train()

        all_probs = []
        with torch.no_grad():
            for _ in range(self.mc_passes):
                probs = F.softmax(self.model(tensor), dim=1)
                all_probs.append(probs.cpu().numpy())

        self.model.eval()

        stacked    = np.stack(all_probs, axis=0)       # (T, 1, 3, H, W)
        mean_probs = stacked.mean(axis=0)[0]           # (3, H, W)
        var_probs  = stacked.var(axis=0)[0]            # (3, H, W)

        disc_mask = (mean_probs[1] > self.disc_threshold).astype(np.uint8)
        cup_mask  = (mean_probs[2] > self.cup_threshold).astype(np.uint8)

        if self.debug:
            cv2.imwrite("debug_disc_prob.png", (mean_probs[1] * 255).astype(np.uint8))
            cv2.imwrite("debug_cup_prob.png",  (mean_probs[2] * 255).astype(np.uint8))
            print(
                f"[debug] threshold disc>{self.disc_threshold} → {disc_mask.sum()} px  |  "
                f"cup>{self.cup_threshold} → {cup_mask.sum()} px"
            )

        return disc_mask, cup_mask, var_probs

    # ──────────────────────────────────────────────────────────────────
    # ROI-restricted uncertainty
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    def _roi_uncertainty(
        var_probs: np.ndarray,
        disc_mask: np.ndarray,
        cup_mask:  np.ndarray,
    ) -> float:
        """
        Variance restricted to disc+cup pixels only.
        Averaging over the whole image buries clinical uncertainty under
        high background confidence (background is ~85 % of the canvas).
        """
        roi = (disc_mask == 1) | (cup_mask == 1)
        if not roi.any():
            return float(var_probs[1:].mean())
        return float(var_probs[1:, roi].mean())

    # ──────────────────────────────────────────────────────────────────
    # public API
    # ──────────────────────────────────────────────────────────────────

    def run(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Correct processing order:

            threshold (raw)
            → enforce containment          ← BEFORE any morphology
            → clean disc  (5×5 kernel)
            → clean cup   (adaptive kernel)
            → enforce containment again    ← morphology can re-expand cup
            → ROI uncertainty
            → minimum-cup sanity gate
            → clinical pipeline
        """
        tensor = self._preprocess(image)

        # Step 1: raw threshold — both masks uncleaned
        disc_raw, cup_raw, var_probs = self._mc_segment(tensor)

        # Step 2: first enforcement on raw masks
        # Cup has NOT been morphologically expanded yet, so violations here
        # are true model misalignments, not artifacts of our own processing.
        disc_mask, cup_mask, violations = self._enforce_cup_in_disc(disc_raw, cup_raw)

        if self.debug:
            print(
                f"[debug] after 1st enforce: disc={disc_mask.sum()} px  "
                f"cup={cup_mask.sum()} px  violations={violations}"
            )

        # Step 3: clean both masks after containment is guaranteed
        disc_mask = self._clean_disc(disc_mask)
        cup_mask  = self._clean_cup(cup_mask)

        if self.debug:
            print(
                f"[debug] after cleanup: disc={disc_mask.sum()} px  "
                f"cup={cup_mask.sum()} px"
            )

        # Step 4: second enforcement — morphological close can slightly
        # expand the cup back outside a clean (potentially shrunk) disc
        disc_mask, cup_mask, extra = self._enforce_cup_in_disc(disc_mask, cup_mask)
        violations += extra

        # Step 5: uncertainty over anatomical ROI only
        uncertainty = self._roi_uncertainty(var_probs, disc_mask, cup_mask)

        # Step 6: minimum-cup sanity gate
        disc_area = int(disc_mask.sum())
        cup_area  = int(cup_mask.sum())
        extra_warnings: list = []

        if disc_area < _MIN_DISC_AREA_PX:
            extra_warnings.append(
                f"Disc too small ({disc_area} px) — image may not be a fundus photo."
            )

        if 0 < cup_area < _MIN_CUP_AREA_PX:
            extra_warnings.append(
                f"Cup remnant too small ({cup_area} px) — suppressed as unreliable."
            )
            cup_mask = np.zeros_like(cup_mask)
            cup_area = 0

        if cup_area > 0 and disc_area > 0 and (cup_area / disc_area) < _MIN_CUP_DISC_RATIO:
            extra_warnings.append(
                f"Cup/disc ratio ({cup_area}/{disc_area} = {cup_area/disc_area:.3f}) "
                "is implausibly low — cup reading may be unreliable."
            )

        # Step 7: clinical pipeline
        clinical = run_clinical_pipeline(
            disc_mask=disc_mask,
            cup_mask=cup_mask,
            uncertainty=uncertainty,
            uncertainty_threshold=self.uncertainty_threshold,
        )

        if violations > 0:
            clinical.warnings.append(
                f"Mask corrected: {violations} cup pixels clipped to disc boundary."
            )
        clinical.warnings.extend(extra_warnings)

        report = {
            'vcdr':             clinical.vcdr,
            'isnt':             clinical.isnt.to_dict(),
            'risk_level':       clinical.risk_level.value,
            'uncertainty':      round(uncertainty, 6),
            'high_uncertainty': clinical.high_uncertainty,
            'disc_area_px':     clinical.disc_area_px,
            'cup_area_px':      clinical.cup_area_px,
            'disc_center':      clinical.disc_center,
            'cup_center':       clinical.cup_center,
            'sanity_passed':    clinical.sanity_passed,
            'warnings':         clinical.warnings,
        }

        return {
            'disc_mask':   disc_mask,
            'cup_mask':    cup_mask,
            'uncertainty': uncertainty,
            'clinical':    clinical,
            'report':      report,
        }

    def run_from_path(self, image_path: str) -> Dict[str, Any]:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Cannot load image: {image_path}")
        return self.run(image)