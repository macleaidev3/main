"""
GlaucoSense - EyeeSEE Detailed Local Inference
===============================================

Uses the author's supplied EyeeSEE v1 Phase 3 pipeline and clinical logic,
while loading a LOCAL .pt/.pth checkpoint instead of downloading weights.

NO:
    - synthetic images
    - synthetic inference data
    - fake probabilities
    - fabricated confidence
    - ESRGAN enhancement
    - image augmentation
    - artificial cropping
    - artificial masks

The script prints:
    - image information
    - checkpoint information
    - device
    - model input/output information
    - MC passes
    - segmentation pixel counts
    - disc/cup area
    - cup/disc area percentage
    - disc/cup centers
    - vCDR
    - disc vertical diameter
    - cup vertical diameter
    - vCDR verification
    - uncertainty
    - high-uncertainty status
    - sanity check
    - ISNT measurements
    - ISNT rule
    - original EyeeSEE risk
    - GlaucoSense 3-stage result
    - warnings

Required files in the SAME FOLDER as this script:

    model.py
    phase3pipeline.py
    clinical_metrics.py
    checkpoint_loader.py

Example checkpoint:

    checkpoint_epoch_057.pt
"""

from pathlib import Path
import sys
import cv2
import numpy as np
import torch


# ============================================================================
# USER CONFIGURATION
# ============================================================================

# --------------------------------------------------------------------------
# REAL FUNDUS IMAGE PATH
# --------------------------------------------------------------------------
IMAGE_PATH = (r"D:\Anurag BPCL WORK\Glaucosense\TASK NO. PRJ-002111\Old EyeeSEE\ESRGAN_Model_Result\1_OD_1.tif")

# Example:
# IMAGE_PATH = (
#     r"D:\Anurag BPCL WORK\Glaucosense\TASK NO. PRJ-002111"
#     r"\Old EyeeSEE\My_EyeeSEE\1_OD_1.tif"
# )


# --------------------------------------------------------------------------
# REAL EYEESEE CHECKPOINT PATH
# --------------------------------------------------------------------------
CHECKPOINT_PATH = (r"D:\Anurag BPCL WORK\Glaucosense\TASK NO. PRJ-002111\Old EyeeSEE\My_EyeeSEE\checkpoints\checkpoint_epoch_057.pt")

# Example:
# CHECKPOINT_PATH = (
#     r"D:\Anurag BPCL WORK\Glaucosense\TASK NO. PRJ-002111"
#     r"\Old EyeeSEE\My_EyeeSEE\checkpoint_epoch_057.pt"
# )


# ============================================================================
# EYEESEE SPACE APPLICATION SETTINGS
# ============================================================================

# The author's actual app.py uses:
#     mc_passes=2
#
# Keep this at 2 if the goal is to reproduce the Space application's
# actual runtime behavior.
MC_PASSES = 2


# The author's actual app.py uses:
#     uncertainty_threshold=0.05
UNCERTAINTY_THRESHOLD = 0.05


# ============================================================================
# DEVICE
# ============================================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================================
# IMPORT THE AUTHOR'S SUPPLIED EYEESEE CODE
# ============================================================================

from model import UNet
from phase3pipeline import Phase3Pipeline
from checkpoint_loader import load_model_weights
from clinical_metrics import calculate_vcdr


# ============================================================================
# PATH VALIDATION
# ============================================================================

def validate_paths() -> None:
    """
    Verify that the real image and real checkpoint exist.

    The script fails instead of generating a substitute result.
    """

    image_path = Path(IMAGE_PATH)
    checkpoint_path = Path(CHECKPOINT_PATH)

    if not image_path.is_file():
        raise FileNotFoundError(
            f"Fundus image not found:\n{image_path}"
        )

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found:\n{checkpoint_path}"
        )

    if checkpoint_path.suffix.lower() not in {".pt", ".pth"}:
        raise ValueError(
            "CHECKPOINT_PATH must point to a .pt or .pth PyTorch checkpoint."
        )


# ============================================================================
# REAL IMAGE LOADING
# ============================================================================

def load_real_fundus_image(image_path: str) -> np.ndarray:
    """
    Load the actual supplied fundus image.

    This follows the image-loading approach used by the author's app.py.

    No preprocessing is performed here beyond decoding the real image.
    The author's Phase3Pipeline performs its own preprocessing.
    """

    image = cv2.imread(image_path)

    # Windows / Unicode-safe fallback.
    if image is None:
        raw = np.fromfile(
            image_path,
            dtype=np.uint8
        )

        image = cv2.imdecode(
            raw,
            cv2.IMREAD_COLOR
        )

    if image is None:
        raise ValueError(
            f"Unable to read fundus image:\n{image_path}"
        )

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(
            f"Expected a 3-channel color fundus image, "
            f"but received shape {image.shape}"
        )

    return image


# ============================================================================
# BUILD LOCAL EYEESEE PIPELINE
# ============================================================================

def build_local_eyesee_pipeline() -> Phase3Pipeline:
    """
    Create the supplied author's Phase3Pipeline without allowing its
    constructor to download a checkpoint from Hugging Face.

    IMPORTANT:

    The actual Phase3Pipeline methods from the author's file are retained.

    We bypass ONLY the constructor because that constructor performs the
    Hugging Face checkpoint download.

    Then we load the real local checkpoint into the author's UNet.
    """

    # ----------------------------------------------------------------------
    # Bypass Phase3Pipeline.__init__()
    #
    # The original __init__ downloads from Hugging Face.
    # We do not want that because we already have the real local checkpoint.
    # ----------------------------------------------------------------------
    pipeline = Phase3Pipeline.__new__(Phase3Pipeline)

    # ----------------------------------------------------------------------
    # SAME SETTINGS USED BY THE AUTHOR'S PIPELINE / SPACE APP
    # ----------------------------------------------------------------------

    pipeline.repo_id = "Nj-1111/EyeeSEE"

    pipeline.mc_passes = MC_PASSES

    pipeline.uncertainty_threshold = (
        UNCERTAINTY_THRESHOLD
    )

    pipeline.device = DEVICE

    pipeline.token = None

    # ----------------------------------------------------------------------
    # EXACT UNET ARCHITECTURE USED BY THE SUPPLIED MODEL.PY
    # ----------------------------------------------------------------------

    pipeline.model = UNet(
        in_channels=1,
        n_classes=3,
        base_filters=64,
        dropout=0.1,
    )

    # ----------------------------------------------------------------------
    # LOAD THE REAL LOCAL CHECKPOINT
    # ----------------------------------------------------------------------

    load_model_weights(
        model=pipeline.model,
        checkpoint_path=str(
            Path(CHECKPOINT_PATH).resolve()
        ),
        device=DEVICE,
    )

    # Explicit inference mode.
    pipeline.model.eval()

    return pipeline


# ============================================================================
# EYEESEE RISK -> GLAUCOSENSE LABEL
# ============================================================================

def map_eyesee_to_glaucosense(
    risk_level: str
) -> str:
    """
    Map the author's risk terminology to the three labels required by
    GlaucoSense.

    EyeeSEE:
        Healthy
        Glaucoma Suspect
        High Risk

    GlaucoSense:
        No Glaucoma
        Likely Glaucoma
        Glaucoma
    """

    mapping = {
        "Healthy": "No Glaucoma",
        "Glaucoma Suspect": "Likely Glaucoma",
        "High Risk": "Glaucoma",
    }

    if risk_level not in mapping:
        raise RuntimeError(
            "Unexpected EyeeSEE risk level returned by the pipeline: "
            f"{risk_level!r}"
        )

    return mapping[risk_level]


# ============================================================================
# PRINT DETAILED RESULT
# ============================================================================

def print_detailed_report(
    image: np.ndarray,
    result: dict
) -> None:
    """
    Print the actual information returned by the EyeeSEE pipeline.

    Nothing printed here is fabricated.
    """

    if "report" not in result:
        raise RuntimeError(
            "EyeeSEE result does not contain a report."
        )

    report = result["report"]

    if not isinstance(report, dict):
        raise RuntimeError(
            "EyeeSEE report is not a dictionary."
        )

    if "disc_mask" not in result:
        raise RuntimeError(
            "EyeeSEE result does not contain the disc mask."
        )

    if "cup_mask" not in result:
        raise RuntimeError(
            "EyeeSEE result does not contain the cup mask."
        )

    disc_mask = result["disc_mask"]
    cup_mask = result["cup_mask"]

    # ----------------------------------------------------------------------
    # Reuse the author's exact vCDR function.
    #
    # This gives us the vertical diameter details used to calculate vCDR.
    # ----------------------------------------------------------------------

    vcdr_verified, vcdr_details = calculate_vcdr(
        disc_mask,
        cup_mask
    )

    # ----------------------------------------------------------------------
    # Classification
    # ----------------------------------------------------------------------

    eyeesee_risk = report["risk_level"]

    glaucosense_result = map_eyesee_to_glaucosense(
        eyeesee_risk
    )

    # ----------------------------------------------------------------------
    # Structural calculations
    # ----------------------------------------------------------------------

    disc_area = int(
        report["disc_area_px"]
    )

    cup_area = int(
        report["cup_area_px"]
    )

    if disc_area > 0:
        cup_disc_percent = (
            cup_area / disc_area
        ) * 100.0
    else:
        cup_disc_percent = 0.0

    # =========================================================================
    # HEADER
    # =========================================================================

    print()
    print("=" * 76)
    print("EYEESEE / GLAUCOSENSE - DETAILED LOCAL INFERENCE")
    print("=" * 76)

    # =========================================================================
    # IMAGE
    # =========================================================================

    print()
    print("IMAGE")
    print("-" * 76)

    print(
        f"Image name         : "
        f"{Path(IMAGE_PATH).name}"
    )

    print(
        f"Image path         : "
        f"{Path(IMAGE_PATH).resolve()}"
    )

    print(
        f"Original resolution: "
        f"{image.shape[1]} x {image.shape[0]}"
    )

    print(
        f"Array shape        : "
        f"{image.shape}"
    )

    print(
        f"Data type          : "
        f"{image.dtype}"
    )

    # =========================================================================
    # CHECKPOINT / MODEL
    # =========================================================================

    print()
    print("MODEL / CHECKPOINT")
    print("-" * 76)

    print(
        f"Checkpoint         : "
        f"{Path(CHECKPOINT_PATH).resolve()}"
    )

    print(
        "Architecture       : "
        "EyeeSEE U-Net"
    )

    print(
        "Input              : "
        "1-channel grayscale"
    )

    print(
        "Input size         : "
        "512 x 512"
    )

    print(
        "Output             : "
        "3-class segmentation"
    )

    print(
        "Class 0            : "
        "Background"
    )

    print(
        "Class 1            : "
        "Optic Disc"
    )

    print(
        "Class 2            : "
        "Optic Cup"
    )

    print(
        "Base filters       : "
        "64"
    )

    print(
        "Dropout            : "
        "0.1"
    )

    print(
        f"Device             : "
        f"{DEVICE}"
    )

    print(
        f"MC passes          : "
        f"{MC_PASSES}"
    )

    print(
        f"Uncertainty thresh.: "
        f"{UNCERTAINTY_THRESHOLD}"
    )

    # =========================================================================
    # SEGMENTATION
    # =========================================================================

    print()
    print("SEGMENTATION / STRUCTURAL MEASUREMENTS")
    print("-" * 76)

    print(
        f"Disc pixels        : "
        f"{disc_area:,}"
    )

    print(
        f"Cup pixels         : "
        f"{cup_area:,}"
    )

    print(
        f"Cup/Disc area      : "
        f"{cup_disc_percent:.2f}%"
    )

    print(
        f"Disc center        : "
        f"{tuple(report['disc_center'])}"
    )

    print(
        f"Cup center         : "
        f"{tuple(report['cup_center'])}"
    )

    # =========================================================================
    # VCDR
    # =========================================================================

    print()
    print("vCDR")
    print("-" * 76)

    print(
        f"vCDR               : "
        f"{report['vcdr']:.4f}"
    )

    print(
        f"Disc vertical diam.: "
        f"{vcdr_details.get('disc_v_diam_px')}"
    )

    print(
        f"Cup vertical diam. : "
        f"{vcdr_details.get('cup_v_diam_px')}"
    )

    print(
        f"vCDR verification  : "
        f"{vcdr_verified:.4f}"
    )

    # =========================================================================
    # UNCERTAINTY
    # =========================================================================

    print()
    print("UNCERTAINTY / SANITY")
    print("-" * 76)

    print(
        f"Uncertainty        : "
        f"{report['uncertainty']:.6f}"
    )

    print(
        f"High uncertainty   : "
        f"{report['high_uncertainty']}"
    )

    print(
        f"Sanity check       : "
        f"{'PASSED' if report['sanity_passed'] else 'FAILED'}"
    )

    # =========================================================================
    # ISNT
    # =========================================================================

    isnt = report["isnt"]

    print()
    print("ISNT RIM MEASUREMENTS")
    print("-" * 76)

    print(
        f"Inferior           : "
        f"{isnt['inferior']:.4f}"
    )

    print(
        f"Superior           : "
        f"{isnt['superior']:.4f}"
    )

    print(
        f"Nasal              : "
        f"{isnt['nasal']:.4f}"
    )

    print(
        f"Temporal           : "
        f"{isnt['temporal']:.4f}"
    )

    print(
        f"ISNT rule          : "
        f"{'SATISFIED' if isnt['rule_satisfied'] else 'VIOLATED'}"
    )

    # =========================================================================
    # CLASSIFICATION
    # =========================================================================

    print()
    print("CLASSIFICATION")
    print("-" * 76)

    print(
        f"EyeeSEE risk       : "
        f"{eyeesee_risk}"
    )

    print(
        f"GlaucoSense result : "
        f"{glaucosense_result}"
    )

    # =========================================================================
    # WARNINGS
    # =========================================================================

    warnings = report.get(
        "warnings",
        []
    )

    print()
    print("WARNINGS")
    print("-" * 76)

    if warnings:
        for warning in warnings:
            print(
                f"- {warning}"
            )
    else:
        print("None")

    # =========================================================================
    # FOOTER
    # =========================================================================

    print()
    print("=" * 76)
    print()


# ============================================================================
# RUN INFERENCE
# ============================================================================

def infer() -> str:
    """
    Execute one genuine EyeeSEE inference.
    """

    # ------------------------------------------------------------------------
    # Verify real paths
    # ------------------------------------------------------------------------

    validate_paths()

    # ------------------------------------------------------------------------
    # Load the real fundus image
    # ------------------------------------------------------------------------

    image = load_real_fundus_image(
        IMAGE_PATH
    )

    # ------------------------------------------------------------------------
    # Load the real model + local checkpoint
    # ------------------------------------------------------------------------

    pipeline = build_local_eyesee_pipeline()

    # ------------------------------------------------------------------------
    # Run the author's actual Phase3Pipeline.run()
    # ------------------------------------------------------------------------

    result = pipeline.run(
        image
    )

    # ------------------------------------------------------------------------
    # Validate result structure
    # ------------------------------------------------------------------------

    if not isinstance(result, dict):
        raise RuntimeError(
            "EyeeSEE pipeline returned an unexpected result type."
        )

    if "report" not in result:
        raise RuntimeError(
            "EyeeSEE pipeline returned no report."
        )

    report = result["report"]

    if not isinstance(report, dict):
        raise RuntimeError(
            "EyeeSEE report is not a valid dictionary."
        )

    if "risk_level" not in report:
        raise RuntimeError(
            "EyeeSEE report contains no risk_level."
        )

    risk_level = report["risk_level"]

    if not isinstance(risk_level, str):
        raise RuntimeError(
            "EyeeSEE risk_level is not a string."
        )

    # ------------------------------------------------------------------------
    # Print the full genuine report
    # ------------------------------------------------------------------------

    print_detailed_report(
        image=image,
        result=result
    )

    # ------------------------------------------------------------------------
    # Final GlaucoSense label
    # ------------------------------------------------------------------------

    return map_eyesee_to_glaucosense(
        risk_level
    )


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":

    try:

        final_label = infer()

        # Easy final line for later integration into GlaucoSense.
        print(
            f"FINAL RESULT: {final_label}"
        )

    except Exception as exc:

        # IMPORTANT:
        #
        # Never convert an inference failure into a fabricated prediction.
        #
        # The program exits with error code 1 instead.
        print(
            f"ERROR: {exc}",
            file=sys.stderr
        )

        sys.exit(1)