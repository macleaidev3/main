"""
FAST HYBRID FUNDUS IMAGE EVALUATOR
==================================

Purpose:
    Compare ORIGINAL fundus images against ESRGAN-processed images.

The evaluator:
    1. Reads images from two user-provided folders.
    2. Matches original and ESRGAN images by filename.
    3. Calculates multiple image-quality measurements.
    4. Calculates ORIGINAL and ESRGAN quality scores.
    5. Compares ESRGAN against the ORIGINAL image.
    6. Calculates an enhancement/safety score.
    7. Decides KEEP or RED_FLAG based ONLY on final score comparison.
    8. Saves:
           original_image_evaluation.csv
           ESRGAN_image_evaluation.csv
    9. Moves rejected ESRGAN images into:
           ESRGAN_FOLDER/red_flagged/

IMPORTANT:
    FINAL DECISION RULE:

        ESRGAN final score > ORIGINAL final score
            -> KEEP / QUALIFIED

        ESRGAN final score <= ORIGINAL final score
            -> RED_FLAG

    Other measurements such as noise increase, color change,
    excessive edge increase, etc. are still calculated and
    recorded, but they DO NOT independently decide whether
    an image is rejected.

    This is an automated image-quality evaluator.
    It does NOT diagnose glaucoma and does NOT claim that
    an image is clinically superior.

Recommended Python:
    Python 3.10 / 3.11

Required packages:
    pip install opencv-python numpy pandas scikit-image tifffile
"""

import os
import shutil
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
from skimage.metrics import structural_similarity


# ======================= USER SETTINGS ===========================
# ------------------------ PUT YOUR ORIGINAL IMAGE FOLDER HERE -------------------------------
ORIGINAL_FOLDER = r"D:\Anurag BPCL WORK\Glaucosense\Evaluation of Fundus Images\original"

# ------------------------PUT YOUR ESRGAN IMAGE FOLDER HERE -----------------------------------
ESRGAN_FOLDER = r"D:\Anurag BPCL WORK\Glaucosense\Evaluation of Fundus Images\my_fundus_images_results"

# ------------------------------------------------------------
# FINAL SCORE THRESHOLD
# IMPORTANT:
# This value is retained from the original code. IT IS NOT USED FOR THE FINAL DECISION.
# The decision is now based ONLY on:
#     ESRGAN final score > ORIGINAL final score
# ------------------------------------------------------------
MIN_FINAL_SCORE = 60.0

# ------------------------------------------------------------
# MINIMUM IMPROVEMENT REQUIRED
# IMPORTANT:
# This value is retained from the original code. IT IS NOT USED FOR THE FINAL DECISION.
# The decision is now based ONLY on:
#     ESRGAN final score > ORIGINAL final score
# ------------------------------------------------------------
MIN_IMPROVEMENT = 0.0

# ------------------------------ ALLOWED IMAGE EXTENSIONS -----------------------------------
IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}

# ============================== GENERAL IMAGE LOADING =====================================
def load_image(path):
    """
    Loads an image safely.
    Returns:
        BGR image as uint8 numpy array
    """

    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise ValueError(f"Could not read image: {path}")

    # -------------------------- Handle grayscale ------------------------------------
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # -------------------------- Handle BGRA ------------------------------------------
    elif (image.ndim == 3  and image.shape[2] == 4):
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    # ------------------------ Convert unusual bit depth to uint8 ------------------------
    if image.dtype != np.uint8:
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        image = image.astype(np.uint8)
    return image

# =========================== FUNDUS REGION DETECTION =====================================
def get_fundus_mask(image):
    """
    Finds the circular/elliptical retinal region.
    Black background around the retina should not dominate
    the quality calculations.  """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)    
    # Blur slightly to remove tiny noise.
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    # Otsu threshold.
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Remove tiny regions.
    kernel = np.ones((7, 7), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,kernel)
    # Find contours.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.ones(gray.shape, dtype=np.uint8) * 255

    # Largest contour is normally the fundus.
    largest = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(largest)
    image_area = (gray.shape[0] *  gray.shape[1])
    # If segmentation is unreasonable, use entire image.
    if area < image_area * 0.05:
        return np.ones(gray.shape, dtype=np.uint8) * 255
    fundus_mask = np.zeros_like(gray)
    cv2.drawContours(fundus_mask, [largest], -1, 255, thickness=cv2.FILLED)
    return fundus_mask

# ========================= IMAGE RESIZING FOR COMPARISON ================================
def prepare_pair(original,  processed):
    """
    Makes the two images the same size.
    We resize the processed image to the original dimensions only for evaluation. We do NOT modify the actual file.
    """
    if original.shape[:2] != processed.shape[:2]:
        processed = cv2.resize(processed, (original.shape[1], original.shape[0]), interpolation=cv2.INTER_AREA)
    return original, processed

# ====================== SHARPNESS ===================================
def calculate_sharpness(image, mask):
    """
    Measures fine detail using Laplacian variance. Higher normally means sharper.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pixels = gray[ mask > 0 ]
    if pixels.size < 100:
        pixels = gray.flatten()
    # Use Laplacian on complete image.
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    values = laplacian[ mask > 0 ]
    if values.size < 100:
        values = laplacian.flatten()
    return float(np.var(values))

# ======================= CONTRAST ===================================
def calculate_contrast(image, mask):
    """
    Measures the spread of brightness values. Higher is not automatically better.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pixels = gray[ mask > 0 ]
    if pixels.size < 100:
        pixels = gray.flatten()
    return float( np.std(pixels))

# ========================= BRIGHTNESS =================================
def calculate_brightness(image, mask):
    """
    Calculates average brightness.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pixels = gray[ mask > 0 ]
    if pixels.size < 100:
        pixels = gray.flatten()
    return float( np.mean(pixels))

# ========================== NOISE ESTIMATION =================================
def calculate_noise(image, mask):
    """
    Estimates high-frequency noise.
    This is deliberately conservative because aggressive sharpening can look like noise.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    smooth = cv2.GaussianBlur(gray, (3, 3), 0)
    residual = (gray.astype(np.float32) -  smooth.astype(np.float32))
    values = residual[ mask > 0 ]
    if values.size < 100:
        values = residual.flatten()
    return float( np.std(values))

# ======================= ENTROPY =====================================
def calculate_entropy(image, mask):
    """
    Measures image information/complexity.
    """
    gray = cv2.cvtColor(image,  cv2.COLOR_BGR2GRAY)
    pixels = gray[ mask > 0]
    if pixels.size < 100:
        pixels = gray.flatten()
    histogram = np.bincount(pixels, minlength=256).astype( np.float64)
    histogram /= histogram.sum()
    histogram = histogram[histogram > 0]
    entropy = -np.sum(histogram * np.log2(histogram))
    return float(entropy)

# ====================== CLIPPING =================================
def calculate_clipping(image, mask):
    """
    Calculates percentage of pixels that are almost black or almost white. Excessive clipping means loss of image information.
    """
    gray = cv2.cvtColor(image,  cv2.COLOR_BGR2GRAY)
    pixels = gray[ mask > 0]
    if pixels.size < 100:
        pixels = gray.flatten()
    dark = np.sum(pixels <= 2)
    bright = np.sum(pixels >= 253)
    clipping = ((dark + bright) / max(len(pixels), 1)) * 100.0
    return float(clipping)

# ======================= DYNAMIC RANGE =====================================
def calculate_dynamic_range(image,  mask):
    """
    Measures the useful brightness range.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pixels = gray[ mask > 0]
    if pixels.size < 100:
        pixels = gray.flatten()
    low = np.percentile(pixels, 1)
    high = np.percentile(pixels, 99)
    return float(high - low)

# ======================== EDGE DENSITY ==================================
def calculate_edge_density(image, mask):
    """
    Measures how much edge information exists. This is useful for detecting excessive sharpening.
    """
    gray = cv2.cvtColor( image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    valid = mask > 0
    if np.sum(valid) < 100:
        valid = np.ones(gray.shape, dtype=bool)
    edge_pixels = np.sum(edges[valid] > 0)
    total_pixels = np.sum(valid)
    if total_pixels == 0:
        return 0.0
    return float(edge_pixels / total_pixels *  100.0)

# =========================== COLOR SATURATION =====================================
def calculate_saturation(image, mask):
    """
    Calculates average HSV saturation.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    pixels = saturation[mask > 0]
    if pixels.size < 100:
        pixels = saturation.flatten()
    return float(np.mean(pixels))

# ========================== COLOR CHANGE =================================
def calculate_color_difference(original, processed, mask):
    """
    Measures average color change between the two images.
    """
    original_lab = cv2.cvtColor(original, cv2.COLOR_BGR2LAB)
    processed_lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
    difference = np.sqrt(np.sum((original_lab.astype(np.float32)  - processed_lab.astype(np.float32)) ** 2, axis=2))
    values = difference[mask > 0]
    if values.size < 100:
        values = difference.flatten()
    return float(np.mean(values))

# ========================== STRUCTURAL SIMILARITY =======================================
def calculate_ssim(original, processed):
    """
    Measures structural similarity.
    1.0 = nearly identical structure
    lower = more structural change
    """
    original_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
    processed_gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    score = structural_similarity(original_gray, processed_gray, data_range=255)
    return float(score)

# ========================== TECHNICAL QUALITY SCORE ======================================
def calculate_technical_score(metrics):
    """
    Converts raw measurements into a 0-100 technical score.The scoring is deliberately based on reasonable ranges rather than simply rewarding maximum sharpness.
    """
    sharpness = metrics["sharpness"]
    contrast = metrics["contrast"]
    brightness = metrics["brightness"]
    noise = metrics["noise"]
    clipping = metrics["clipping"]
    dynamic_range = metrics["dynamic_range"]
    entropy = metrics["entropy"]
    edge_density = metrics["edge_density"]

    # ----------------------- SHARPNESS ---------------------------------------
    sharpness_score = np.clip(np.log1p(sharpness)  / np.log1p(8000) * 100, 0, 100)

    # ----------------------------- CONTRAST -----------------------------------
    contrast_score = np.clip(100 - abs(contrast - 52)  * 1.8, 0, 100)

    # -------------------------- BRIGHTNESS -------------------------------------
    brightness_score = np.clip(100 - abs(brightness - 105) * 1.2, 0, 100)

    # ------------------------- NOISE -------------------------------------------
    noise_score = np.clip(100 -  noise * 5.0,  0, 100)

    # -------------------------- CLIPPING ---------------------------------------
    clipping_score = np.clip(100 -  clipping * 12.0, 0, 100)

    # -------------------------- DYNAMIC RANGE ------------------------------------
    dynamic_score = np.clip(dynamic_range / 2.0, 0, 100)

    # ------------------------ ENTROPY ----------------------------------------
    entropy_score = np.clip((entropy - 4.5) / 3.0 * 100, 0, 100)

    # ------------------------ EDGE DENSITY -------------------------------------
    if edge_density <= 12:
        edge_score = 100
    elif edge_density <= 25:
        edge_score = (100 - (edge_density - 12) * 3.0)
    else:
        edge_score = (61  -  (edge_density - 25) * 4.0)
    edge_score = np.clip(edge_score, 0, 100)

    # -------------------------- FINAL TECHNICAL SCORE -----------------------------------
    score = (sharpness_score * 0.20 + contrast_score * 0.15 + brightness_score * 0.10  + noise_score * 0.15 +
        clipping_score * 0.10  +  dynamic_score * 0.10  + entropy_score * 0.10 + edge_score * 0.10)
    return float(np.clip(score, 0, 100))

# ======================= ENHANCEMENT SAFETY SCORE ===================================
def calculate_enhancement_safety(original_metrics, processed_metrics, ssim_score, color_difference):
    """
    Determines whether ESRGAN improved the image without making suspiciously large changes.
    IMPORTANT:
        This score contributes to the ESRGAN FINAL SCORE.
        However, individual safety measurements such as noise increase or color change DO NOT independently reject the image.
    """
    # ------------------------ 1. Noise change ------------------------------------
    original_noise = (original_metrics["noise"])
    processed_noise = (processed_metrics["noise"])
    noise_ratio = (processed_noise /  max(original_noise, 1e-6))
    if noise_ratio <= 1.10:
        noise_safety = 100
    elif noise_ratio <= 1.50:
        noise_safety = (100 -  (noise_ratio - 1.10) * 120)
    else:
        noise_safety = (52 -  (noise_ratio - 1.50) * 70)
    noise_safety = np.clip(noise_safety, 0, 100)

    # ------------------------ 2. Edge increase --------------------------------------
    original_edges = (original_metrics["edge_density"])
    processed_edges = (processed_metrics["edge_density"])
    edge_ratio = (processed_edges / max(original_edges, 0.1))
    if edge_ratio <= 1.40:
        edge_safety = 100
    elif edge_ratio <= 2.0:
        edge_safety = (100 - (edge_ratio - 1.40) * 100)
    else:
        edge_safety = (40 - (edge_ratio - 2.0) * 60)
    edge_safety = np.clip(edge_safety,  0,  100)

    # --------------------- 3. Structural preservation -------------------------------------
    structural_score = np.clip((ssim_score - 0.70) / 0.30  * 100, 0, 100)

    # -------------------------- 4. Color preservation -------------------------------------
    color_score = np.clip(100 - color_difference * 2.0, 0, 100)

    # -------------------------- 5. Excessive clipping --------------------------------------
    original_clip = (original_metrics["clipping"])
    processed_clip = (processed_metrics["clipping"])
    clipping_increase = (processed_clip  -  original_clip)
    if clipping_increase <= 0.5:
        clipping_safety = 100
    elif clipping_increase <= 3:
        clipping_safety = (100 - (clipping_increase - 0.5) * 30)
    else:
        clipping_safety = (25 - (clipping_increase - 3) * 10)
    clipping_safety = np.clip(clipping_safety, 0, 100)

    # ------------------------ FINAL SAFETY SCORE ---------------------------------
    safety_score = (structural_score * 0.35 + noise_safety * 0.20 + edge_safety * 0.20 + color_score * 0.15 + clipping_safety * 0.10)
    return float(np.clip(safety_score, 0, 100))

# ========================== IMAGE METRIC COLLECTION ===============================
def evaluate_image(image):
    """
    Calculates all basic metrics for one image.
    """
    mask = get_fundus_mask(image)
    metrics = {
        "sharpness": calculate_sharpness(image,mask),
        "contrast":  calculate_contrast(image, mask),
        "brightness": calculate_brightness(image, mask),
        "noise": calculate_noise(image, mask),
        "entropy": calculate_entropy(image, mask),
        "clipping": calculate_clipping(image, mask),
        "dynamic_range": calculate_dynamic_range(image, mask),
        "edge_density": calculate_edge_density(image,mask),
        "saturation": calculate_saturation(image, mask),
    }
    metrics["technical_score"] = (calculate_technical_score(metrics))
    return metrics

# ============================= FIND IMAGES ======================================
def get_images(folder):
    """
    Returns all supported image files in a folder. red_flagged is deliberately ignored because it is a
    subfolder and only files directly inside the folder are considered.
    """

    folder = Path(folder)
    images = []
    for path in folder.iterdir():
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        images.append(path)
    return sorted(images, key=lambda p: p.name.lower())

# ======================= FILE MATCHING =====================================
def build_file_dictionary(paths):
    """
    Creates:
        filename -> file path
    """
    return {
        path.name.lower(): path for path in paths
    }

# ========================== PROCESS ONE PAIR ==================================
def process_pair(original_path, processed_path):
    """
    Evaluates one original/ESRGAN pair.
    """
    original = load_image(original_path)
    processed = load_image(processed_path)
    original, processed = prepare_pair(original, processed)

    # -------------------------- Evaluate both independently. -----------------------------
    original_metrics = evaluate_image(original)
    processed_metrics = evaluate_image(processed)

    # -------------------------- Compare structures. ---------------------------------
    ssim_score = calculate_ssim(original, processed)
    mask = get_fundus_mask(original)
    color_difference = (calculate_color_difference(original, processed, mask ))

    # ------------------------------ Safety score. -----------------------------------
    safety_score = (calculate_enhancement_safety(original_metrics, processed_metrics, ssim_score, color_difference ))

    # ---------------------------- Technical improvement. -----------------------------
    improvement = (processed_metrics["technical_score"]  -  original_metrics["technical_score"])

    # ============================= FINAL SCORE CALCULATION ============================

    # --------------------------------------------------------
    # ORIGINAL FINAL SCORE
    # The original image does not have an enhancement/safety calculation. Therefore its final score is its technical score.
    # --------------------------------------------------------
    original_final_score = (original_metrics["technical_score"])

    # --------------------------------------------------------
    # ESRGAN FINAL SCORE
    # SAME calculation as the original code.
    # Technical quality = 40%
    # Safety            = 60%
    # --------------------------------------------------------
    esrgan_final_score = (processed_metrics["technical_score"] * 0.40 + safety_score * 0.60)

    # ========================================================
    # FINAL DECISION
    # THIS IS THE ONLY THING CHANGED.
    # ESRGAN final score > Original final score
    #       -> KEEP    #
    # ESRGAN final score <= Original final score
    #       -> RED_FLAG
    # ========================================================

    if (esrgan_final_score >  original_final_score):
        decision = "KEEP"
        decision_reason = ("ESRGAN_FINAL_SCORE_GREATER_THAN_ORIGINAL")
    else:
        decision = "RED_FLAG"
        decision_reason = ("ESRGAN_FINAL_SCORE_NOT_GREATER_THAN_ORIGINAL")

    # --------------------------------------------------------
    # Additional measurements are STILL calculated. They are recorded for analysis but do NOT override the final decision.
    # --------------------------------------------------------

    additional_reasons = []
    if esrgan_final_score < MIN_FINAL_SCORE:
        additional_reasons.append("FINAL_SCORE_BELOW_60")
    if improvement < MIN_IMPROVEMENT:
        additional_reasons.append("TECHNICAL_SCORE_NOT_IMPROVED")
    if safety_score < 45:
        additional_reasons.append("LOW_SAFETY_SCORE")
    if (processed_metrics["noise"]  >  original_metrics["noise"] * 1.75):
        additional_reasons.append("NOISE_INCREASE")
    if (processed_metrics["edge_density"]  > original_metrics["edge_density"] * 2.0):
        additional_reasons.append("EXCESSIVE_EDGE_INCREASE")
    if color_difference > 20:
        additional_reasons.append("LARGE_COLOR_CHANGE")

    # ========================= ORIGINAL CSV RECORD ====================================
    original_record = {
        "filename": original_path.name,
        "sharpness": original_metrics["sharpness"],
        "contrast":  original_metrics["contrast"],
        "brightness": original_metrics["brightness"],
        "noise": original_metrics["noise"],
        "entropy": original_metrics["entropy"],
        "clipping": original_metrics["clipping"],
        "dynamic_range": original_metrics["dynamic_range"],
        "edge_density": original_metrics["edge_density"],
        "saturation": original_metrics["saturation"],
        "technical_score": original_metrics["technical_score"],
        "final_evaluation_score": original_final_score,
    }

    # ======================== ESRGAN CSV RECORD =======================================
    processed_record = {
        "filename": processed_path.name,
        "sharpness": processed_metrics["sharpness"],
        "contrast":  processed_metrics["contrast"],
        "brightness": processed_metrics["brightness"],
        "noise": processed_metrics["noise"],
        "entropy": processed_metrics["entropy"],
        "clipping": processed_metrics["clipping"],
        "dynamic_range": processed_metrics["dynamic_range"],
        "edge_density": processed_metrics["edge_density"],
        "saturation": processed_metrics["saturation"],
        "technical_score": processed_metrics["technical_score"],
        "ssim_vs_original": ssim_score,
        "color_difference_vs_original":  color_difference,
        "enhancement_safety_score": safety_score,
        "improvement_vs_original":  improvement,
        "original_final_score": original_final_score,
        "final_evaluation_score": esrgan_final_score,
        "final_score_difference": (esrgan_final_score- original_final_score),
        "decision": decision,
        "decision_reason": decision_reason,
        "additional_warnings": ";".join(additional_reasons)
            if additional_reasons else "NONE",
    }

    return (original_record,processed_record)

# ========================= MAIN ==================================
def main():
    original_folder = Path(ORIGINAL_FOLDER)
    esrgan_folder = Path(ESRGAN_FOLDER)

    # ---------------------------- Check folders. ------------------------
    if not original_folder.exists():
        raise FileNotFoundError(f"Original folder does not exist:\n {original_folder}")
    if not esrgan_folder.exists():
        raise FileNotFoundError(f"ESRGAN folder does not exist:\n {esrgan_folder}")

    # --------------------------- Create red_flagged folder.------------------------------------------
    red_flagged_folder = (esrgan_folder  / "red_flagged")
    red_flagged_folder.mkdir( exist_ok=True)

    # ------------------------- Find images. ------------------------------------
    original_images = get_images(original_folder)
    esrgan_images = get_images(esrgan_folder)
    print()
    print("=" * 70)
    print("FAST HYBRID FUNDUS IMAGE EVALUATOR")
    print("=" * 70)
    print(f"Original images found : {len(original_images)}")
    print(f"ESRGAN images found   : {len(esrgan_images)}")

    # ---------------------------- Match files by filename. --------------------------------------------------------
    original_dict = (build_file_dictionary(original_images))
    esrgan_dict = (build_file_dictionary(esrgan_images))
    common_names = sorted(set(original_dict.keys()) & set(esrgan_dict.keys()))
    missing_original = sorted(set(esrgan_dict.keys()) - set(original_dict.keys()))
    missing_esrgan = sorted(set(original_dict.keys()) - set(esrgan_dict.keys()))
    print(f"Matched image pairs   : {len(common_names)}")
    print(f"Missing original      : {len(missing_original)}")
    print(f"Missing ESRGAN        : {len(missing_esrgan)}")
    print("=" * 70)
    if not common_names:
        print("\nERROR: No matching image pairs found.")
        print("\nThe filenames must match between the two folders.")
        return

    # ----------------------------- Results. -------------------------------
    original_results = []
    esrgan_results = []
    keep_count = 0
    red_flag_count = 0
    error_count = 0

    # --------------------------- Process each image. ----------------------------
    total = len(common_names)
    for index, filename in enumerate(common_names, start=1):
        original_path = (original_dict[filename])
        esrgan_path = (esrgan_dict[filename])
        print()
        print(f"[{index}/{total}] {filename}")
        try:
            (original_record,processed_record) = process_pair(original_path, esrgan_path)
            original_results.append(original_record)
            esrgan_results.append(processed_record)
            decision = (processed_record["decision"])
            original_score = (processed_record["original_final_score"])
            esrgan_score = (processed_record["final_evaluation_score"])
            difference = (processed_record["final_score_difference"])

            # -------------------------- PRINT FINAL SCORES ------------------------------------------------
            print(f"    Original final score : {original_score:.2f}")
            print(f"    ESRGAN final score   : {esrgan_score:.2f}")
            print(f"    Score difference     : {difference:+.2f}")
            print(f"    Decision             : {decision}")

            # ----------------------- COUNT ------------------------------------------------
            if decision == "KEEP":
                keep_count += 1
            else:
                red_flag_count += 1

                # --------------------- MOVE rejected ESRGAN image. --------------------------------------------
                destination = ( red_flagged_folder / esrgan_path.name )
                # Avoid accidental overwrite.
                if destination.exists():
                    stem = (destination.stem)
                    suffix = (destination.suffix)
                    counter = 1

                    while destination.exists():
                        destination = (red_flagged_folder / f"{stem}_{counter}{suffix}")
                        counter += 1
                shutil.move(str(esrgan_path), str(destination))

                print("    Moved to red_flagged/")
        except Exception as exc:
            error_count += 1
            print(f"    ERROR: {exc}")

    # =========================== SAVE ORIGINAL CSV =======================================
    original_csv = (original_folder  / "original_image_evaluation.csv" )
    original_dataframe = pd.DataFrame(original_results)
    original_dataframe.to_csv(original_csv,  index=False)

    # =========================== SAVE ESRGAN CSV =========================================
    esrgan_csv = (esrgan_folder  /  "ESRGAN_image_evaluation.csv")
    esrgan_dataframe = pd.DataFrame( esrgan_results)
    esrgan_dataframe.to_csv( esrgan_csv, index=False)

    # ========================= FINAL SUMMARY ==============================================
    print()
    print("=" * 70)
    print("EVALUATION COMPLETE")
    print("=" * 70)
    print(f"Total matched pairs : {total}")
    print(f"KEEP                 : {keep_count}")
    print(f"RED FLAG             : {red_flag_count}")
    print(f"ERROR                : {error_count}")
    print()
    print( "FINAL DECISION RULE:")
    print("ESRGAN final score > "
        "Original final score = KEEP"
    )
    print(
        "ESRGAN final score <= "
        "Original final score = RED_FLAG"
    )
    print()
    print("Original CSV:" )
    print(original_csv)
    print()
    print( "ESRGAN CSV:")
    print( esrgan_csv )
    print()
    print("Red-flagged images:")
    print(red_flagged_folder)
    print("=" * 70)

# ==================== PROGRAM START =====================================
if __name__ == "__main__":
    main()
