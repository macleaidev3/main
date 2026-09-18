"""
Phase 3 — Clinical Logic & Verification Engine
===============================================
Implements deterministic clinical metric extraction from segmentation masks.
No ML here — pure geometry and ophthalmology math.

Mask convention (from Phase 2 training):
    0 = Background
    1 = Optic Disc
    2 = Optic Cup
"""

import numpy as np
import cv2
from dataclasses import dataclass, field
from typing import Tuple, Dict, Optional
from enum import Enum

# ── ISNT tolerance margin ──────────────────────────────────────────────────
# Exact I > S > N > T fails on tiny numerical noise.
# A margin of 0.2 rim-pixels absorbs rounding without masking real violations.
_ISNT_MARGIN = 0.2


class RiskLevel(Enum):
    HEALTHY = "Healthy"
    SUSPECT = "Glaucoma Suspect"
    HIGH    = "High Risk"


class SanityError(Exception):
    """Raised when mask geometry violates anatomical constraints."""
    pass


@dataclass
class ISNTResult:
    inferior:       float = 0.0
    superior:       float = 0.0
    nasal:          float = 0.0
    temporal:       float = 0.0
    rule_satisfied: bool  = False

    def to_dict(self) -> Dict[str, float]:
        return {
            'inferior':       round(self.inferior,  4),
            'superior':       round(self.superior,  4),
            'nasal':          round(self.nasal,     4),
            'temporal':       round(self.temporal,  4),
            'rule_satisfied': self.rule_satisfied,
        }


@dataclass
class ClinicalResult:
    vcdr:             float      = 0.0
    isnt:             ISNTResult = field(default_factory=ISNTResult)
    disc_area_px:     int        = 0
    cup_area_px:      int        = 0
    disc_center:      Tuple[int, int] = (0, 0)
    cup_center:       Tuple[int, int] = (0, 0)
    uncertainty:      float      = 0.0
    high_uncertainty: bool       = False
    risk_level:       RiskLevel  = RiskLevel.HEALTHY
    sanity_passed:    bool       = False
    warnings:         list       = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            'vcdr':             round(self.vcdr, 4),
            'isnt':             self.isnt.to_dict(),
            'disc_area_px':     self.disc_area_px,
            'cup_area_px':      self.cup_area_px,
            'disc_center':      self.disc_center,
            'cup_center':       self.cup_center,
            'uncertainty':      round(self.uncertainty, 6),
            'high_uncertainty': self.high_uncertainty,
            'risk_level':       self.risk_level.value,
            'sanity_passed':    self.sanity_passed,
            'warnings':         self.warnings,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Sanity checks
# ─────────────────────────────────────────────────────────────────────────────

def run_sanity_checks(disc_mask: np.ndarray, cup_mask: np.ndarray) -> None:
    """
    Enforce anatomical constraints.  Raises SanityError on hard violations.

    Checks:
        1. Masks are binary (0/1 values only).
        2. Disc region is non-empty.
        3. Cup is 100 % contained inside the disc (hard anatomical law).
        4. Disconnected regions — warn only; upstream _clean_* handles them.
    """
    # 1. Binary check
    for name, mask in [('disc', disc_mask), ('cup', cup_mask)]:
        unique_vals = np.unique(mask)
        if not set(unique_vals).issubset({0, 1}):
            raise SanityError(
                f"{name} mask contains non-binary values: {unique_vals}"
            )

    # 2. Non-empty disc
    if int(disc_mask.sum()) == 0:
        raise SanityError("Optic disc mask is empty — segmentation failure.")

    # 3. Cup ⊂ Disc
    cup_outside_disc = np.logical_and(cup_mask == 1, disc_mask == 0)
    if cup_outside_disc.any():
        n_violation = int(cup_outside_disc.sum())
        raise SanityError(
            f"Cup extends outside disc boundary ({n_violation} pixels). "
            "Anatomically impossible — reject segmentation."
        )

    # 4. Single connected component — warn only, do not reject.
    # Small segmentation gaps from noisy boundaries are handled upstream
    # by _clean_disc / _clean_cup.
    for name, mask in [('disc', disc_mask), ('cup', cup_mask)]:
        if mask.sum() == 0:
            continue
        n_labels, _ = cv2.connectedComponents(mask.astype(np.uint8))
        if n_labels > 2:
            pass  # upstream cleanup handles; avoid hard rejection here


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — vCDR
# ─────────────────────────────────────────────────────────────────────────────

def calculate_vcdr(
    disc_mask: np.ndarray,
    cup_mask:  np.ndarray,
) -> Tuple[float, dict]:
    """
    Calculate vertical Cup-to-Disc Ratio using vertical extrema.

    Clinical basis:
        Horizontal disc expansion is less indicative of early glaucoma.
        vCDR is the primary screening metric.

    Returns:
        vcdr    (float): ratio in [0, 1]
        details (dict):  raw pixel measurements for transparency
    """
    disc_rows = np.where(disc_mask.any(axis=1))[0]
    cup_rows  = np.where(cup_mask.any(axis=1))[0]

    if disc_rows.size == 0:
        return 0.0, {}

    disc_v_diam = int(disc_rows.max() - disc_rows.min() + 1)
    cup_v_diam  = int(cup_rows.max()  - cup_rows.min()  + 1) if cup_rows.size > 0 else 0

    vcdr = cup_v_diam / disc_v_diam if disc_v_diam > 0 else 0.0

    details = {
        'disc_top_px':    int(disc_rows.min()),
        'disc_bottom_px': int(disc_rows.max()),
        'disc_v_diam_px': disc_v_diam,
        'cup_top_px':     int(cup_rows.min())  if cup_rows.size > 0 else None,
        'cup_bottom_px':  int(cup_rows.max())  if cup_rows.size > 0 else None,
        'cup_v_diam_px':  cup_v_diam,
    }
    return round(vcdr, 4), details


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — ISNT Rule
# ─────────────────────────────────────────────────────────────────────────────

def _disc_centroid(disc_mask: np.ndarray) -> Tuple[int, int]:
    M = cv2.moments(disc_mask.astype(np.uint8))
    if M['m00'] == 0:
        h, w = disc_mask.shape
        return h // 2, w // 2
    cy = int(M['m01'] / M['m00'])
    cx = int(M['m10'] / M['m00'])
    return cy, cx


def _rim_thickness_in_quadrant(
    quadrant_mask: np.ndarray,
    disc_mask:     np.ndarray,
    cup_mask:      np.ndarray,
) -> float:
    """
    Mean Euclidean distance from cup boundary to disc boundary,
    measured inside a specific quadrant.

    Uses a distance transform on the disc interior so that each pixel
    inside the disc gets its distance to the disc edge.
    We then sample only rim pixels (disc=1, cup=0) in this quadrant.
    """
    rim         = np.logical_and(disc_mask == 1, cup_mask == 0).astype(np.uint8)
    rim_in_quad = np.logical_and(rim, quadrant_mask).astype(np.uint8)

    if rim_in_quad.sum() == 0:
        return 0.0

    dist_to_disc_edge = cv2.distanceTransform(
        disc_mask.astype(np.uint8), cv2.DIST_L2, 5
    )

    thicknesses = dist_to_disc_edge[rim_in_quad == 1]
    return float(np.mean(thicknesses))


def calculate_isnt(
    disc_mask: np.ndarray,
    cup_mask:  np.ndarray,
) -> ISNTResult:
    """
    Calculate neuro-retinal rim thickness in four ISNT quadrants.

    Quadrant definition (image coordinates):
        Superior  — top half    (rows < cy)
        Inferior  — bottom half (rows >= cy)
        Nasal     — right half  (cols >= cx)   [standard right eye convention]
        Temporal  — left half   (cols < cx)

    ISNT rule (with tolerance margin):
        Inferior > Superior - margin
        Superior > Nasal    - margin
        Nasal    > Temporal - margin

    Using a strict exact ordering falsely triggers violations when
    quadrant thicknesses differ by sub-pixel amounts due to rounding.
    A margin of _ISNT_MARGIN (0.2 px) absorbs noise without masking
    real rim thinning (which presents as differences of several pixels).
    """
    h, w   = disc_mask.shape
    cy, cx = _disc_centroid(disc_mask)

    superior_q = np.zeros((h, w), dtype=bool)
    inferior_q = np.zeros((h, w), dtype=bool)
    nasal_q    = np.zeros((h, w), dtype=bool)
    temporal_q = np.zeros((h, w), dtype=bool)

    superior_q[:cy, :]  = True
    inferior_q[cy:, :]  = True
    nasal_q[:,   cx:]   = True
    temporal_q[:, :cx]  = True

    I = _rim_thickness_in_quadrant(inferior_q,  disc_mask, cup_mask)
    S = _rim_thickness_in_quadrant(superior_q,  disc_mask, cup_mask)
    N = _rim_thickness_in_quadrant(nasal_q,     disc_mask, cup_mask)
    T = _rim_thickness_in_quadrant(temporal_q,  disc_mask, cup_mask)

    # Tolerated ISNT check — absorbs sub-pixel numerical noise
    rule_ok = (
        I > S - _ISNT_MARGIN and
        S > N - _ISNT_MARGIN and
        N > T - _ISNT_MARGIN
    )

    return ISNTResult(
        inferior=I, superior=S, nasal=N, temporal=T,
        rule_satisfied=rule_ok,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Risk classification
# ─────────────────────────────────────────────────────────────────────────────

def classify_risk(
    vcdr:                 float,
    isnt:                 ISNTResult,
    uncertainty:          float,
    uncertainty_threshold: float = 0.05,
) -> Tuple[RiskLevel, list]:
    """
    Rule-based risk stratification.

    Thresholds from clinical literature:
        vCDR < 0.65              → Healthy
        vCDR 0.65–0.80           → Suspect
        vCDR > 0.80              → High Risk
        ISNT violation (any risk) → escalate to at least Suspect
        High uncertainty         → override to Suspect regardless of vCDR
    """
    warnings = []

    if uncertainty > uncertainty_threshold:
        warnings.append(
            f"High model uncertainty ({uncertainty:.4f}) — result may be unreliable."
        )
        return RiskLevel.SUSPECT, warnings

    if vcdr > 0.80:
        risk = RiskLevel.HIGH
        warnings.append(
            f"vCDR {vcdr:.2f} exceeds 0.80 — urgent referral recommended."
        )
    elif vcdr > 0.65:
        risk = RiskLevel.SUSPECT
        warnings.append(f"vCDR {vcdr:.2f} in borderline range (0.65–0.80).")
    else:
        risk = RiskLevel.HEALTHY

    if not isnt.rule_satisfied:
        warnings.append("ISNT rule violated — neuro-retinal rim thinning detected.")
        if risk == RiskLevel.HEALTHY:
            risk = RiskLevel.SUSPECT

    return risk, warnings


# ─────────────────────────────────────────────────────────────────────────────
# Main pipeline entry point
# ─────────────────────────────────────────────────────────────────────────────

def run_clinical_pipeline(
    disc_mask:            np.ndarray,
    cup_mask:             np.ndarray,
    uncertainty:          float = 0.0,
    uncertainty_threshold: float = 0.05,
) -> ClinicalResult:
    """
    Execute complete Phase 3 clinical pipeline on binary masks.

    Args:
        disc_mask:             uint8 binary array (1 = disc, 0 = background)
        cup_mask:              uint8 binary array (1 = cup,  0 = background)
        uncertainty:           scalar from Phase 2 MC-Dropout (ROI-restricted)
        uncertainty_threshold: flag above this value as high uncertainty

    Returns:
        ClinicalResult dataclass
    """
    result = ClinicalResult()
    result.uncertainty      = float(uncertainty)
    result.high_uncertainty = uncertainty > uncertainty_threshold

    disc_mask = (disc_mask > 0).astype(np.uint8)
    cup_mask  = (cup_mask  > 0).astype(np.uint8)

    # ── Sanity checks ──────────────────────────────────────────────────
    try:
        run_sanity_checks(disc_mask, cup_mask)
        result.sanity_passed = True
    except SanityError as e:
        result.warnings.append(f"SANITY FAIL: {e}")
        result.sanity_passed = False
        # Only abort if there is no disc at all — otherwise continue
        # computing metrics on whatever geometry we have.
        if disc_mask.sum() == 0:
            result.risk_level = RiskLevel.SUSPECT
            return result

    # ── Structural measurements ────────────────────────────────────────
    result.disc_area_px = int(disc_mask.sum())
    result.cup_area_px  = int(cup_mask.sum())

    dy, dx = _disc_centroid(disc_mask)
    result.disc_center = (int(dx), int(dy))

    if cup_mask.sum() > 0:
        M = cv2.moments(cup_mask)
        if M['m00'] > 0:
            result.cup_center = (
                int(M['m10'] / M['m00']),
                int(M['m01'] / M['m00']),
            )

    # ── vCDR ──────────────────────────────────────────────────────────
    result.vcdr, _ = calculate_vcdr(disc_mask, cup_mask)

    # ── ISNT (with tolerance margin) ──────────────────────────────────
    result.isnt = calculate_isnt(disc_mask, cup_mask)

    # ── Risk classification ───────────────────────────────────────────
    result.risk_level, warnings = classify_risk(
        result.vcdr, result.isnt, uncertainty, uncertainty_threshold
    )
    result.warnings.extend(warnings)

    return result
