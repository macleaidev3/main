import cv2 ## importing Open source Computer Vision
import numpy as np
import os
import pandas as pd
import re


# ============================================================
# 1. SHARPNESS
# ============================================================

def calculate_sharpness(image):
    """
    Measures image sharpness using Laplacian variance.

    Higher value = sharper image
    Lower value  = blurrier image
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)## This line is converting the image to grayscale. Earlier an image might be colorful so
    ##it will be intended as "2048 X 2048 X 3" where 2048 means hight, 2048 mean width and 3 means thress different type of colors, so the "gray" line the image becomes "2048 X 2048" which mean the image is having only gray color and not other colors.

    laplacian = cv2.Laplacian(gray, cv2.CV_64F) ## Laplacian is a mathematical operator that measures second-order changes in image intensity. In this line of code we have "cv2.CV_64F", this tells the OpenCV which numerical data type to use for the result.    
    ## in this line  "64F" means 64-bit floating point.

    variance = laplacian.var()## Laplacian Variance is a statistical measurement used in computer vision to determine whether an image is sharp or blurry.

    return variance

'''
In 'def calculate_sharpness(image)' the function is doing-"Take the fundus image, convert it to grayscale, detect how strongly the pixel 
intensities change using the Laplacian, measure how much those responses vary, and use that variance as an indicator of image sharpness.". 
The reason we are scaling the image in grey is because the Laplacian sharpness calculation primarily needs intensity changes, not color
information.    
'''

# ============================================================
# 2. CONTRAST
# ============================================================

def calculate_contrast(image):
    """
    Measures contrast using the standard deviation of grayscale pixel intensities.
    Higher value generally means stronger contrast.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    contrast = gray.std()

    return contrast

'''
In 'def calculte_contrast(image)' the function is doing- "Take the fundus image, convert it to grayscale, look at the brightness value of every
pixcel, calculate how much those brightness values vary from one another, and use that variation as the image's contrast measurement"
'''


# ============================================================
# 3. BRIGHTNESS
# ============================================================

def calculate_brightness(image):
    """
    Calculates average grayscale brightness.

    0   = black
    255 = white
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    brightness = gray.mean()

    return brightness


# ============================================================
# 4. CLIPPING
# ============================================================

def calculate_clipping(image):
    """
    Measures how many pixels are extremely dark or
    extremely bright.

    Excessive clipping can indicate poor exposure.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    total_pixels = gray.size

    dark_pixels = np.sum(gray <= 5)

    bright_pixels = np.sum(gray >= 250)

    clipped_pixels = dark_pixels + bright_pixels

    clipping_percentage = (
        clipped_pixels / total_pixels
    ) * 100

    return clipping_percentage


# ============================================================
# 5. NOISE ESTIMATION
# ============================================================

def calculate_noise(image):
    """
    Estimates high-frequency noise using the difference
    between the original image and a lightly blurred image.

    Higher value = potentially more noise
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blurred = cv2.GaussianBlur(
        gray,
        (3, 3),
        0
    )

    noise = (
        gray.astype(np.float32)
        - blurred.astype(np.float32)
    )

    noise_level = np.std(noise)

    return noise_level


# ============================================================
# 6. FUNDUS REGION DETECTION
# ============================================================

def calculate_fundus_coverage(image):
    """
    Estimates how much of the image contains the illuminated
    fundus region rather than the black background.

    This is a simple approximation.
    """

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Pixels above this threshold are considered
    # part of the visible fundus region.
    mask = gray > 10

    coverage = mask.mean() * 100

    return coverage


# ============================================================
# 7. NORMALIZATION FUNCTIONS
# ============================================================

def normalize_sharpness(value):

    minimum = 50
    maximum = 1500

    score = (value - minimum) / (maximum - minimum)

    return np.clip(score, 0, 1) * 100


def normalize_contrast(value):

    minimum = 15
    maximum = 80

    score = (value - minimum) / (maximum - minimum)

    return np.clip(score, 0, 1) * 100


def normalize_brightness(value):

    """
    Penalizes images that are extremely dark or bright.

    Ideal brightness is approximately around the middle.
    """

    ideal = 120

    maximum_distance = 120

    distance = abs(value - ideal)

    score = 1 - (distance / maximum_distance)

    return np.clip(score, 0, 1) * 100


def normalize_clipping(value):

    """
    Less clipping = better quality.
    """

    maximum = 20

    score = 1 - (value / maximum)

    return np.clip(score, 0, 1) * 100


def normalize_noise(value):

    maximum = 30

    score = 1 - (value / maximum)

    return np.clip(score, 0, 1) * 100


def normalize_coverage(value):

    minimum = 30
    maximum = 100

    score = (value - minimum) / (maximum - minimum)

    return np.clip(score, 0, 1) * 100


# ============================================================
# 8. FINAL QUALITY SCORE
# ============================================================

def calculate_quality_score(image):

    # --------------------------------------------------------
    # Raw measurements
    # --------------------------------------------------------

    sharpness = calculate_sharpness(image)

    contrast = calculate_contrast(image)

    brightness = calculate_brightness(image)

    clipping = calculate_clipping(image)

    noise = calculate_noise(image)

    coverage = calculate_fundus_coverage(image)


    # --------------------------------------------------------
    # Convert measurements to 0-100 scores
    # --------------------------------------------------------

    sharpness_score = normalize_sharpness(sharpness)

    contrast_score = normalize_contrast(contrast)

    brightness_score = normalize_brightness(brightness)

    clipping_score = normalize_clipping(clipping)

    noise_score = normalize_noise(noise)

    coverage_score = normalize_coverage(coverage)


    # --------------------------------------------------------
    # Weighted final score
    # --------------------------------------------------------

    final_score = (
        0.30 * sharpness_score +
        0.20 * contrast_score +
        0.15 * brightness_score +
        0.10 * clipping_score +
        0.10 * noise_score +
        0.15 * coverage_score
    )


    return {

        "final_score":
            round(final_score, 2),

        "sharpness_raw":
            round(sharpness, 2),

        "sharpness_score":
            round(sharpness_score, 2),

        "contrast_raw":
            round(contrast, 2),

        "contrast_score":
            round(contrast_score, 2),

        "brightness_raw":
            round(brightness, 2),

        "brightness_score":
            round(brightness_score, 2),

        "clipping_percentage":
            round(clipping, 2),

        "clipping_score":
            round(clipping_score, 2),

        "noise_raw":
            round(noise, 2),

        "noise_score":
            round(noise_score, 2),

        "fundus_coverage":
            round(coverage, 2),

        "coverage_score":
            round(coverage_score, 2)
    }


# ============================================================
# 9. PROCESS WHOLE IMAGE FOLDER
# ============================================================

def evaluate_image_folder(folder_path, output_csv):

    """
    Reads all supported images from a folder,
    calculates image quality scores,
    and saves the results to a CSV file.
    """

    # --------------------------------------------------------
    # Supported image formats
    # --------------------------------------------------------

    supported_extensions = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".tif",
        ".tiff"
    )


    # --------------------------------------------------------
    # Get image filenames
    # --------------------------------------------------------

    image_files = [
        filename
        for filename in os.listdir(folder_path)
        if filename.lower().endswith(supported_extensions)
    ]

    # Natural ascending order
    image_files.sort(
        key=lambda filename: [
            int(text) if text.isdigit() else text.lower()
            for text in re.split(r'(\d+)', filename)
        ]
    )


    if len(image_files) == 0:

        print("No supported images found in the folder.")

        return


    print(f"Found {len(image_files)} images.")

    print("Starting image quality evaluation...\n")


    # --------------------------------------------------------
    # Store results here
    # --------------------------------------------------------

    results = []


    # --------------------------------------------------------
    # Process every image
    # --------------------------------------------------------

    for index, filename in enumerate(image_files, start=1):

        image_path = os.path.join(
            folder_path,
            filename
        )


        print(
            f"Processing {index}/{len(image_files)}: "
            f"{filename}"
        )


        # Read image
        image = cv2.imread(image_path)


        # Check whether image was successfully read
        if image is None:

            print(
                f"WARNING: Could not read {filename}"
            )

            continue


        # Calculate quality metrics
        quality = calculate_quality_score(image)


        # Add filename to the result
        row = {
            "Image_file_name": filename,
            **quality
        }


        # Store result
        results.append(row)


    # --------------------------------------------------------
    # Create DataFrame
    # --------------------------------------------------------

    df = pd.DataFrame(results)


    # --------------------------------------------------------
    # Arrange columns in desired order
    # --------------------------------------------------------

    columns = [
        "Image_file_name",

        "sharpness_raw",
        "sharpness_score",

        "contrast_raw",
        "contrast_score",

        "brightness_raw",
        "brightness_score",

        "clipping_percentage",
        "clipping_score",

        "noise_raw",
        "noise_score",

        "fundus_coverage",
        "coverage_score",

        "final_score"
    ]


    df = df[columns]


    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    df.to_csv(
        output_csv,
        index=False
    )


    print("\n======================================")
    print("IMAGE QUALITY EVALUATION COMPLETE")
    print("======================================")

    print(f"Images successfully evaluated: {len(df)}")

    print(f"CSV saved at:")
    print(output_csv)

    print("======================================")


# ============================================================
# 10. MAIN
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Folder containing fundus images
    # --------------------------------------------------------

    image_folder = (
        r"D:\Anurag BPCL WORK\Glaucosense\Evaluation of Fundus Images\my_fundus_images")


    # --------------------------------------------------------
    # Output CSV location
    # --------------------------------------------------------

    output_csv = (
        r"D:\Anurag BPCL WORK\Glaucosense\Evaluation of Fundus Images\fundus_image_quality_scores.csv")


    # --------------------------------------------------------
    # Evaluate entire folder
    # --------------------------------------------------------

    evaluate_image_folder(
        image_folder,
        output_csv
    )