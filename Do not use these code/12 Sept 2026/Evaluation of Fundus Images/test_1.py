"""
============================================================
IMAGE QUALITY IMPROVEMENT EVALUATOR
============================================================

Purpose:
    Compare ORIGINAL images against ESRGAN-processed images.

Output:
    A CSV file containing:

        File Name
        Original Quality Score
        ESRGAN Quality Score
        Quality Change (%)
        Verdict
        Quality Statement

Examples:

    Image quality improved by 30.00%.

    Image quality improved by 0.50%.

    Image quality remained 0% — no measurable improvement.

    Image quality decreased by 8.00% — ESRGAN degraded the image.

IMPORTANT:
    This script does NOT use MANIQA or any previously downloaded
    image-quality model.

============================================================
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path


# ============================================================
# 1. USER SETTINGS
# ============================================================

# Folder containing ORIGINAL images
ORIGINAL_FOLDER = Path(
    r"D:\Anurag BPCL WORK\Glaucosense\Evaluation of Fundus Images\Original_images"
)

# Folder containing ESRGAN processed images
ESRGAN_FOLDER = Path(
    r"D:\Anurag BPCL WORK\Glaucosense\Evaluation of Fundus Images\ESRGAN_Result"
)

# CSV output location
OUTPUT_CSV = Path(
    r"D:\Anurag BPCL WORK\Glaucosense\Evaluation of Fundus Images\ESRGAN_Result\ESRGAN_Quality_Evaluation.csv"
)


# ============================================================
# 2. SUPPORTED IMAGE FORMATS
# ============================================================

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp"
}


# ============================================================
# 3. CHECK FOLDERS
# ============================================================

if not ORIGINAL_FOLDER.exists():
    raise FileNotFoundError(
        f"Original image folder not found:\n{ORIGINAL_FOLDER}"
    )

if not ESRGAN_FOLDER.exists():
    raise FileNotFoundError(
        f"ESRGAN image folder not found:\n{ESRGAN_FOLDER}"
    )


# ============================================================
# 4. FIND IMAGES
# ============================================================

original_images = {
    image.name.lower(): image
    for image in ORIGINAL_FOLDER.iterdir()
    if image.is_file()
    and image.suffix.lower() in SUPPORTED_EXTENSIONS
}

esrgan_images = {
    image.name.lower(): image
    for image in ESRGAN_FOLDER.iterdir()
    if image.is_file()
    and image.suffix.lower() in SUPPORTED_EXTENSIONS
}


# ============================================================
# 5. MATCH ORIGINAL AND ESRGAN IMAGES
# ============================================================

matched_files = sorted(
    set(original_images.keys())
    &
    set(esrgan_images.keys())
)


if not matched_files:
    raise RuntimeError(
        "No matching filenames were found between the "
        "Original and ESRGAN folders."
    )


print("=" * 70)
print("IMAGE QUALITY EVALUATION")
print("=" * 70)

print(f"Original images : {len(original_images)}")
print(f"ESRGAN images   : {len(esrgan_images)}")
print(f"Matched pairs   : {len(matched_files)}")
print()


# ============================================================
# 6. IMAGE QUALITY MEASUREMENTS
# ============================================================

def calculate_sharpness(gray_image):
    """
    Calculates sharpness using the variance of the
    Laplacian.

    Higher value = sharper image.
    """

    return cv2.Laplacian(
        gray_image,
        cv2.CV_64F
    ).var()


def calculate_contrast(gray_image):
    """
    Calculates global contrast using standard deviation.

    Higher value = greater contrast.
    """

    return float(
        np.std(gray_image)
    )


def calculate_entropy(gray_image):
    """
    Calculates image entropy.

    Higher entropy generally indicates more
    information/detail variation in the image.
    """

    histogram = cv2.calcHist(
        [gray_image],
        [0],
        None,
        [256],
        [0, 256]
    )

    histogram = histogram.flatten()

    histogram = histogram[
        histogram > 0
    ]

    probability = (
        histogram /
        histogram.sum()
    )

    entropy = -np.sum(
        probability *
        np.log2(probability)
    )

    return float(entropy)


# ============================================================
# 7. CALCULATE QUALITY SCORE
# ============================================================

def calculate_quality_score(image):
    """
    Creates a normalized quality score from:

        - Sharpness
        - Contrast
        - Entropy

    The score is used consistently for both images.

    IMPORTANT:
        This is a relative image-quality index, not a
        universal absolute image-quality measurement.
    """

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    # --------------------------------------------------------
    # Sharpness
    # --------------------------------------------------------

    sharpness = calculate_sharpness(gray)

    # --------------------------------------------------------
    # Contrast
    # --------------------------------------------------------

    contrast = calculate_contrast(gray)

    # --------------------------------------------------------
    # Entropy
    # --------------------------------------------------------

    entropy = calculate_entropy(gray)

    return {
        "sharpness": sharpness,
        "contrast": contrast,
        "entropy": entropy
    }


# ============================================================
# 8. NORMALIZE TWO IMAGES
# ============================================================

def calculate_relative_score(
    original_metrics,
    esrgan_metrics
):
    """
    Calculates relative quality components.

    Each ESRGAN metric is compared with the
    corresponding ORIGINAL metric.
    """

    original_sharpness = (
        original_metrics["sharpness"]
    )

    esrgan_sharpness = (
        esrgan_metrics["sharpness"]
    )

    original_contrast = (
        original_metrics["contrast"]
    )

    esrgan_contrast = (
        esrgan_metrics["contrast"]
    )

    original_entropy = (
        original_metrics["entropy"]
    )

    esrgan_entropy = (
        esrgan_metrics["entropy"]
    )


    # --------------------------------------------------------
    # Prevent division by zero
    # --------------------------------------------------------

    if original_sharpness > 0:
        sharpness_ratio = (
            esrgan_sharpness /
            original_sharpness
        )
    else:
        sharpness_ratio = 1.0


    if original_contrast > 0:
        contrast_ratio = (
            esrgan_contrast /
            original_contrast
        )
    else:
        contrast_ratio = 1.0


    if original_entropy > 0:
        entropy_ratio = (
            esrgan_entropy /
            original_entropy
        )
    else:
        entropy_ratio = 1.0


    # --------------------------------------------------------
    # Combined relative quality
    # --------------------------------------------------------

    quality_ratio = (
        0.40 * sharpness_ratio
        +
        0.30 * contrast_ratio
        +
        0.30 * entropy_ratio
    )


    return quality_ratio


# ============================================================
# 9. PROCESS IMAGES
# ============================================================

results = []


for index, filename in enumerate(
    matched_files,
    start=1
):

    original_path = (
        original_images[filename]
    )

    esrgan_path = (
        esrgan_images[filename]
    )


    print(
        f"[{index}/{len(matched_files)}] "
        f"{original_path.name}"
    )


    try:

        # ----------------------------------------------------
        # READ ORIGINAL
        # ----------------------------------------------------

        original = cv2.imread(
            str(original_path)
        )

        if original is None:
            raise ValueError(
                "Could not read original image."
            )


        # ----------------------------------------------------
        # READ ESRGAN
        # ----------------------------------------------------

        esrgan = cv2.imread(
            str(esrgan_path)
        )

        if esrgan is None:
            raise ValueError(
                "Could not read ESRGAN image."
            )


        # ----------------------------------------------------
        # CALCULATE METRICS
        # ----------------------------------------------------

        original_metrics = (
            calculate_quality_score(
                original
            )
        )

        esrgan_metrics = (
            calculate_quality_score(
                esrgan
            )
        )


        # ----------------------------------------------------
        # RELATIVE QUALITY
        # ----------------------------------------------------

        quality_ratio = (
            calculate_relative_score(
                original_metrics,
                esrgan_metrics
            )
        )


        # ----------------------------------------------------
        # CONVERT TO PERCENTAGE CHANGE
        # ----------------------------------------------------

        quality_change_percent = (
            quality_ratio - 1.0
        ) * 100


        # ----------------------------------------------------
        # CLASSIFY RESULT
        # ----------------------------------------------------

        # Tiny numerical differences are treated as
        # effectively unchanged.

        if abs(quality_change_percent) < 0.01:

            verdict = "NO CHANGE"

            statement = (
                "Image quality remained 0% "
                "— no measurable improvement."
            )

            quality_change_percent = 0.0


        elif quality_change_percent > 0:

            verdict = "IMPROVED"

            statement = (
                f"Image quality improved by "
                f"{quality_change_percent:.2f}%."
            )


        else:

            verdict = "DEGRADED"

            decrease = abs(
                quality_change_percent
            )

            statement = (
                f"Image quality decreased by "
                f"{decrease:.2f}% "
                f"— ESRGAN degraded the image."
            )


        # ----------------------------------------------------
        # QUALITY SCORES
        # ----------------------------------------------------

        # Original baseline = 100
        #
        # ESRGAN score represents its quality relative
        # to the original.

        original_quality_score = 100.0

        esrgan_quality_score = (
            quality_ratio * 100
        )


        # ----------------------------------------------------
        # STORE RESULT
        # ----------------------------------------------------

        results.append({

            "File Name":
                original_path.name,

            "Original Quality Score":
                round(
                    original_quality_score,
                    4
                ),

            "ESRGAN Quality Score":
                round(
                    esrgan_quality_score,
                    4
                ),

            "Quality Change (%)":
                round(
                    quality_change_percent,
                    4
                ),

            "Verdict":
                verdict,

            "Quality Statement":
                statement,

            "Original Sharpness":
                round(
                    original_metrics["sharpness"],
                    4
                ),

            "ESRGAN Sharpness":
                round(
                    esrgan_metrics["sharpness"],
                    4
                ),

            "Original Contrast":
                round(
                    original_metrics["contrast"],
                    4
                ),

            "ESRGAN Contrast":
                round(
                    esrgan_metrics["contrast"],
                    4
                ),

            "Original Entropy":
                round(
                    original_metrics["entropy"],
                    4
                ),

            "ESRGAN Entropy":
                round(
                    esrgan_metrics["entropy"],
                    4
                )

        })


        # ----------------------------------------------------
        # PRINT RESULT
        # ----------------------------------------------------

        print(
            f"    Result: {statement}"
        )

        print()


    except Exception as error:

        print(
            f"    ERROR: {error}"
        )

        results.append({

            "File Name":
                original_path.name,

            "Original Quality Score":
                None,

            "ESRGAN Quality Score":
                None,

            "Quality Change (%)":
                None,

            "Verdict":
                "ERROR",

            "Quality Statement":
                str(error),

            "Original Sharpness":
                None,

            "ESRGAN Sharpness":
                None,

            "Original Contrast":
                None,

            "ESRGAN Contrast":
                None,

            "Original Entropy":
                None,

            "ESRGAN Entropy":
                None

        })


# ============================================================
# 10. CREATE CSV
# ============================================================

df = pd.DataFrame(results)


OUTPUT_CSV.parent.mkdir(
    parents=True,
    exist_ok=True
)


df.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8-sig"
)


# ============================================================
# 11. FINAL SUMMARY
# ============================================================

valid_results = df[
    df["Verdict"].isin(
        [
            "IMPROVED",
            "DEGRADED",
            "NO CHANGE"
        ]
    )
]


improved = (
    valid_results["Verdict"]
    == "IMPROVED"
).sum()


degraded = (
    valid_results["Verdict"]
    == "DEGRADED"
).sum()


unchanged = (
    valid_results["Verdict"]
    == "NO CHANGE"
).sum()


print("=" * 70)
print("EVALUATION FINISHED")
print("=" * 70)

print(
    f"Images evaluated : {len(valid_results)}"
)

print(
    f"Improved         : {improved}"
)

print(
    f"Degraded         : {degraded}"
)

print(
    f"No change        : {unchanged}"
)

print()

print(
    f"CSV saved to:\n{OUTPUT_CSV}"
)

print("=" * 70)