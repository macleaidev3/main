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
from clinical_metrics import run_clinical_pipeline, ClinicalResult


class Phase3Pipeline:

    def __init__(
        self,
        repo_id: str = "Nj-1111/EyeeSEE",
        epoch: Optional[int] = None,
        mc_passes: int = 20,
        uncertainty_threshold: float = 0.05,
        device: Optional[torch.device] = None,
        token: Optional[str] = None
    ):
        self.repo_id               = repo_id
        self.mc_passes             = mc_passes
        self.uncertainty_threshold = uncertainty_threshold

        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

        self.token = (
            token or
            os.getenv('HF_TOKEN_2') or
            os.getenv('HF_TOKEN')
        )

        #IMP: lower dropout improves stability
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

    # ──────────────────────────────────────────────────────────────────────
    # preprocessing
    # ──────────────────────────────────────────────────────────────────────

    def _preprocess(self, image: np.ndarray) -> torch.Tensor:

        if image.ndim == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        image = cv2.resize(
            image,
            (512, 512),
            interpolation=cv2.INTER_AREA
        )

        image = image.astype(np.float32) / 255.0

        return (
            torch.from_numpy(image)
            .unsqueeze(0)
            .unsqueeze(0)
            .to(self.device)
        )

    # ──────────────────────────────────────────────────────────────────────
    # mask cleanup
    # ──────────────────────────────────────────────────────────────────────

    def _clean_mask(self, binary_mask: np.ndarray) -> np.ndarray:

        binary_mask = binary_mask.astype(np.uint8)
        if np.sum(binary_mask) == 0:
            return binary_mask                          # Avoid processing empty masks

        kernel = np.ones((5, 5), np.uint8)

        # Remove speckle noise
        binary_mask = cv2.morphologyEx(
            binary_mask,
            cv2.MORPH_OPEN,
            kernel
        )

        # Fill holes
        binary_mask = cv2.morphologyEx(
            binary_mask,
            cv2.MORPH_CLOSE,
            kernel
        )

        # Keep only largest connected component
        n, labels, stats, _ = cv2.connectedComponentsWithStats(binary_mask)

        if n <= 1:
            return binary_mask

        largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])

        cleaned = np.zeros_like(binary_mask)
        cleaned[labels == largest] = 1

        return cleaned

    # ──────────────────────────────────────────────────────────────────────
    # MC dropout segmentation
    # ──────────────────────────────────────────────────────────────────────

    def _mc_segment(
        self,
        tensor: torch.Tensor
    ) -> Tuple[np.ndarray, np.ndarray, float]:


        # Keep BatchNorm frozen
        self.model.eval()


        # Re-enable ONLY dropout layers
        for m in self.model.modules():
            if isinstance(m, (torch.nn.Dropout)):        #,torch.nn.Dropout2d, torch.nn.Dropout3d
                m.train()


        all_probs = []

        with torch.no_grad():
            for _ in range(self.mc_passes):
                logits = self.model(tensor)

                probs = F.softmax(logits, dim=1)

                all_probs.append(probs.cpu().numpy())



        # Restore eval mode
        self.model.eval()

        all_probs = np.stack(all_probs, axis=0)

        mean_probs = all_probs.mean(axis=0)[0]
        var_probs = all_probs.var(axis=0)[0]
    
        # Lower threshold for debugging
        disc_mask = (mean_probs[1] > 0.25).astype(np.uint8)
        cup_mask = (mean_probs[2] > 0.25).astype(np.uint8)
   

        # TEMP: disable cleanup during debugging
        disc_mask = self._clean_mask(disc_mask)
        cup_mask = self._clean_mask(cup_mask)

        return (
            disc_mask,
            cup_mask,
            float(var_probs[1:].mean())
        )

    # ──────────────────────────────────────────────────────────────────────
    # anatomical correction
    # ──────────────────────────────────────────────────────────────────────

    def _enforce_cup_in_disc(
        self,
        disc_mask: np.ndarray,
        cup_mask: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, int]:

        corrected_cup = np.logical_and(
            cup_mask == 1,
            disc_mask == 1
        ).astype(np.uint8)

        violations = (
            int(cup_mask.sum()) -
            int(corrected_cup.sum())
        )

        return disc_mask, corrected_cup, violations

    # ──────────────────────────────────────────────────────────────────────
    # public API
    # ──────────────────────────────────────────────────────────────────────

    def run(self, image: np.ndarray) -> Dict[str, Any]:

        tensor = self._preprocess(image)

        disc_mask, cup_mask, uncertainty = self._mc_segment(tensor)

        disc_mask, cup_mask, violations = self._enforce_cup_in_disc(
            disc_mask,
            cup_mask
        )
        #cup_mask = self._clean_mask(cup_mask)

        clinical = run_clinical_pipeline(
            disc_mask=disc_mask,
            cup_mask=cup_mask,
            uncertainty=uncertainty,
            uncertainty_threshold=self.uncertainty_threshold
        )

        if violations > 0:
            clinical.warnings.append(
                f"Mask corrected: {violations} cup pixels clipped to disc boundary."
            )

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
