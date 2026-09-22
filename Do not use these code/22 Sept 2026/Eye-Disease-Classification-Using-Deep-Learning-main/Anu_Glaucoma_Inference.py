import os
# ============================================================
# IMPORTANT: Set these BEFORE importing TensorFlow / Keras.
# ============================================================
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["KERAS_BACKEND"] = "tensorflow"

# ==================== IMPORTS =======================================
import json
import sys
from pathlib import Path
import cv2
import numpy as np
import tensorflow as tf
import keras

# ========================= USER PATHS ======================================

MODEL_PATH = (r"D:\Anurag BPCL WORK\Glaucosense\TASK NO. PRJ-002111\Eye-Disease-Classification-Using-Deep-Learning-main\output\models\NASNetMobile_best.keras")

IMAGE_SOURCE = (r"D:\Anurag BPCL WORK\Glaucosense\TASK NO. PRJ-002111\Eye-Disease-Classification-Using-Deep-Learning-main\ESRGAN_Model_Result\5_OS_2.tif")

# =================== MODEL SETTINGS ============================================================
IMAGE_SIZE = 224
SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}
# ====================== START ============================================================
print()
print("=" * 80)
print("GLAUCOSENSE - NASNETMOBILE 4-CLASS EYE DISEASE INFERENCE")
print("=" * 80)
print()
print("Python executable:")
print(sys.executable)
print()
print("TensorFlow:")
print(tf.__version__)
print()
print("Keras:")
print(keras.__version__)
print()
print("Keras backend:")
print(keras.backend.backend())

# ========================= VERIFY PATHS===================================================
model_path = Path(MODEL_PATH)
image_source = Path(IMAGE_SOURCE)

if not model_path.exists():
    raise FileNotFoundError(f"\nMODEL FILE NOT FOUND:\n {model_path}")

if model_path.suffix.lower() != ".keras":
    raise ValueError(f"\nThe model must be a Keras .keras file:\n {model_path}")

if not image_source.exists():
    raise FileNotFoundError(f"\nIMAGE SOURCE NOT FOUND:\n {image_source}")

# =========================== CLASS MAPPING ===============================================
output_dir = model_path.parent.parent
class_indices_path = output_dir / "class_indices.json"

if not class_indices_path.exists():
    raise FileNotFoundError(f"\nclass_indices.json was not found.\n\n Expected:\n {class_indices_path}")

with open(class_indices_path, "r", encoding="utf-8") as f:
    class_indices = json.load(f)

print()
print("=" * 80)
print("CLASS MAPPING")
print("=" * 80)
print("class_indices:", class_indices)

# ======================== VERIFY REQUIRED CLASSES ====================================
required_classes = {
    "cataract",
    "diabetic_retinopathy",
    "glaucoma",
    "normal",
}

if not required_classes.issubset(class_indices.keys()):
    raise RuntimeError(f"\nclass_indices.json does not contain the expected four classes.\n\n Found:\n{class_indices}")

CLASS_BY_INDEX = {
    int(index): class_name for class_name, index in class_indices.items()
}

# ------------------------- Verify indices are unique and contiguous -----------------------------------------
expected_indices = set(range(4))
if set(CLASS_BY_INDEX.keys()) != expected_indices:
    raise RuntimeError(f"\nUnexpected class indices.\n\n Found indices:\n {sorted(CLASS_BY_INDEX.keys())}\n Expected:\n {sorted(expected_indices)}")
CATARACT_INDEX = int(class_indices["cataract"])
DR_INDEX = int(class_indices["diabetic_retinopathy"])
GLAUCOMA_INDEX = int(class_indices["glaucoma"])
NORMAL_INDEX = int(class_indices["normal"])

print("Cataract index:", CATARACT_INDEX)
print("Diabetic Retinopathy index:", DR_INDEX)
print("Glaucoma index:", GLAUCOMA_INDEX)
print("Normal index:", NORMAL_INDEX)

# ===================== NATIVE KERAS 3 MODEL LOADING ============================================================
print()
print("=" * 80)
print("LOADING NATIVE KERAS 3 MODEL")
print("=" * 80)
print()
print("Model file:")
print(model_path)
print()
print("Using keras.saving.load_model()")
print("No manual HDF5 weight extraction.")
print("No manual weight-to-layer assignment.")

# ------------------------------------------------------------
# Load the COMPLETE saved model compile=False is correct because we only need inference.
# ------------------------------------------------------------

keras.backend.clear_session()
try:
    model = keras.saving.load_model(model_path, compile=False, safe_mode=True)
except Exception as exc:
    raise RuntimeError(
        "\nFAILED TO LOAD THE .keras MODEL NATIVELY.\n\n"
        "This script requires a Keras 3 compatible environment.\n"
        f"Model:\n{model_path}\n\n Original error:\n{exc}"
    ) from exc

print()
print("Native .keras model loaded successfully.")

# ============================= MODEL STRUCTURE ==========================================
print()
print("=" * 80)
print("LOADED MODEL STRUCTURE")
print("=" * 80)
print()
print("Input shape :", model.input_shape)
print("Output shape:", model.output_shape)

# ========================= VERIFY MODEL STRUCTURE ========================================
if tuple(model.input_shape) != (None,  IMAGE_SIZE, IMAGE_SIZE, 3):
    raise RuntimeError(f"\nUnexpected model input shape.\n Expected: (None, {IMAGE_SIZE}, {IMAGE_SIZE}, 3)\n Found   : {model.input_shape}")
if tuple(model.output_shape) != (None, 4):
    raise RuntimeError(f"\nUnexpected model output shape.\n Expected: (None, 4)\n Found   : {model.output_shape}")

# ============================== GET IMAGE LIST ===========================================
def get_images(source: Path):
    if source.is_file():
        if (source.suffix.lower() not in SUPPORTED_EXTENSIONS):
            raise ValueError(f"\nUnsupported image format:\n {source}")
        return [source]

    if source.is_dir():
        images = sorted([p  for p in source.iterdir() if (p.is_file()  and p.suffix.lower() in SUPPORTED_EXTENSIONS)])
        if not images:
            raise FileNotFoundError(f"\nNo supported images found:\n {source}")
        return images

    raise FileNotFoundError(f"\nInvalid image source:\n {source}")
images = get_images(image_source)
print()
print("=" * 80)
print(f"Images found: {len(images)}")

# ========================== REMOVE BLACK BORDER =======================================
def remove_black_border(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    coords = np.column_stack(np.where(threshold > 0))

    if coords.size == 0:
        return image

    x, y, w, h = cv2.boundingRect(coords)
    cropped = image[
        y:y + h,
        x:x + w
    ]
    if cropped.size == 0:
        return image
    return cropped


# ============================ PREPROCESS IMAGE =====================================
def preprocess_image(image_path):
    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)

    if image is None:
        raise RuntimeError(f"\nUnable to read image:\n {image_path}")
    original_height = image.shape[0]
    original_width = image.shape[1]

    # ------------------------ GRAYSCALE ---------------------------------------------
    if image.ndim == 2:
        # Deterministically duplicate grayscale into 3 identical channels. No synthetic colorization.
        image = np.stack([image, image, image],  axis=-1)
        image_type = "GRAYSCALE"
    # ------------------------- SINGLE CHANNEL --------------------------------------------------------
    elif (image.ndim == 3   and  image.shape[2] == 1):
        image = np.repeat(image, 3,  axis=2)
        image_type = "GRAYSCALE"
    # ----------------------- FOUR CHANNEL -------------------------------------------
    elif (image.ndim == 3  and  image.shape[2] == 4):
        image = cv2.cvtColor(image,  cv2.COLOR_BGRA2BGR)
        image_type = "COLOUR"
    # ------------------------- THREE CHANNEL ----------------------------------------
    elif (image.ndim == 3  and image.shape[2] == 3):
        image_type = "COLOUR"
    else:
        raise RuntimeError(f"\nUnsupported image shape:\n {image.shape}")

    # ------------------------------- DATA TYPE --------------------------------------
    if image.dtype == np.uint8:
        pass
    elif image.dtype == np.uint16:
        image = (image.astype(np.float32) / 257.0).clip(0, 255).astype(np.uint8)
    else:
        raise RuntimeError(f"\nUnsupported image dtype:\n {image.dtype}")
    # ------------------------ BLACK BORDER ----------------------------------------
    image = remove_black_border(image)
    # ------------------------ RESIZE --------------------------------------------
    image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    # ------------------------ BGR -> RGB --------------------------------------------
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # --------------------------------------------------------
    # NORMALIZATION
    # IMPORTANT: This preserves the /255 preprocessing used by the repository's evaluation pipeline.
    # --------------------------------------------------------

    image = (image.astype(np.float32) / 255.0)

    # ------------------------- BATCH DIMENSION --------------------------------------------
    image = np.expand_dims(image, axis=0)
    print()
    print(f"Original image : {original_width} x {original_height}")
    print(f"Image type     : {image_type}")
    print(f"Resolution     : {original_width} x {original_height}")
    print(f"Model input    : {image.shape}")

    return image
# ========================== INFERENCE ========================================================
print()
print("=" * 80)
print("RUNNING INFERENCE")
print("=" * 80)

for number, image_path in enumerate(images, start=1):
    print()
    print("-" * 80)
    print(f"IMAGE {number}/{len(images)}")
    print("-" * 80)
    print(f"File: {image_path.name}")

    # -------------------------- PREPROCESS ----------------------------------------------
    image = preprocess_image(image_path)

    # --------------------------------------------------------
    # PREDICTION
    # Direct inference.
    # No augmentation.
    # No synthetic data.
    # No ESRGAN processing.
    # --------------------------------------------------------
    prediction = (model(image, training=False).numpy()[0])
    prediction = np.asarray(prediction, dtype=np.float64)

    # ------------------------- VALIDATE OUTPUT --------------------------------------------------------
    if prediction.shape != (4,):
        raise RuntimeError(f"\nUnexpected prediction shape:\n {prediction.shape}")
    if not np.all(np.isfinite(prediction)):
        raise RuntimeError("\nModel produced invalid numerical output.")

    # --------------------------------------------------------
    # CHECK SOFTMAX PROBABILITIES. Do NOT artificially renormalize. We want to report the actual model output.
    # --------------------------------------------------------
    if np.any(prediction < 0) or np.any(prediction > 1):
        raise RuntimeError("\nModel output is outside [0, 1].\n Expected a 4-class softmax output.")
    probability_sum = float(np.sum(prediction))

    if not np.isclose(probability_sum, 1.0, atol=1e-4):
        raise RuntimeError(f"\nModel output does not sum to 1.\n Probability sum: {probability_sum:.8f}")

    # ============================== CLASS PROBABILITIES ========================================================
    cataract_probability = float(prediction[CATARACT_INDEX])
    dr_probability = float(prediction[DR_INDEX])     
    glaucoma_probability = float(prediction[GLAUCOMA_INDEX])
    normal_probability = float(prediction[NORMAL_INDEX])

    # ============================ TOP CLASS ========================================================
    top_index = int(np.argmax(prediction))
    final_result = CLASS_BY_INDEX[top_index]
    model_confidence = float(prediction[top_index])
    # ========================== RESULT ========================================================
    print()
    print("=" * 80)
    print("RESULT")
    print("=" * 80)
    print(f"Cataract probability             : {cataract_probability:.2%}")
    print(f"Diabetic Retinopathy probability  : {dr_probability:.2%}")
    print(f"Glaucoma probability             : {glaucoma_probability:.2%}")
    print(f"Normal probability               : {normal_probability:.2%}" )
    print()
    print(f"FINAL RESULT         : {final_result.title()}")
    print(f"Model confidence     : {model_confidence:.2%}")
    print("=" * 80)
# ============================= COMPLETE ============================================================
print()
print("=" * 80)
print("INFERENCE COMPLETE")
print("=" * 80)