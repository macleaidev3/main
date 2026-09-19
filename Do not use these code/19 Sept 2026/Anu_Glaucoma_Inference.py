import os

# ============================================================
# IMPORTANT:
# Set this BEFORE importing TensorFlow.
# ============================================================

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


# ============================================================
# IMPORTS
# ============================================================

import json
import sys
import zipfile
import tempfile
import shutil
from pathlib import Path

import cv2
import h5py
import numpy as np

import tensorflow as tf
from tensorflow import keras


# ============================================================
# USER PATHS
# ============================================================

MODEL_PATH = (
    r"D:\Anurag BPCL WORK\Glaucosense\TASK NO. PRJ-002111\Eye-Disease-Classification-Using-Deep-Learning-main\output\models\NASNetMobile_best.keras"
)


IMAGE_SOURCE = (
    r"D:\Anurag BPCL WORK\Glaucosense\TASK NO. PRJ-002111\Eye-Disease-Classification-Using-Deep-Learning-main\ESRGAN_Model_Result\4_OD_1_green.tif"
)


# ============================================================
# MODEL SETTINGS
# ============================================================

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


# ============================================================
# START
# ============================================================

print()
print("=" * 80)
print("GLAUCOSENSE - NASNETMOBILE GLAUCOMA / NORMAL INFERENCE")
print("=" * 80)


print()
print("Python executable:")
print(sys.executable)


print()
print("TensorFlow:")
print(tf.__version__)


# ============================================================
# VERIFY PATHS
# ============================================================

model_path = Path(MODEL_PATH)
image_source = Path(IMAGE_SOURCE)


if not model_path.exists():

    raise FileNotFoundError(
        "\nMODEL FILE NOT FOUND:\n"
        f"{model_path}"
    )


if not image_source.exists():

    raise FileNotFoundError(
        "\nIMAGE SOURCE NOT FOUND:\n"
        f"{image_source}"
    )


# ============================================================
# CLASS MAPPING
# ============================================================
#
# The repository creates:
#
# output/class_indices.json
#
# We use the actual file rather than guessing.
# ============================================================

output_dir = model_path.parent.parent

class_indices_path = (
    output_dir / "class_indices.json"
)


if not class_indices_path.exists():

    raise FileNotFoundError(
        "\nclass_indices.json was not found.\n\n"
        f"Expected:\n{class_indices_path}"
    )


with open(
    class_indices_path,
    "r",
    encoding="utf-8"
) as f:

    class_indices = json.load(f)


print()
print("=" * 80)
print("CLASS MAPPING")
print("=" * 80)


print(
    "class_indices:",
    class_indices
)


# ------------------------------------------------------------
# Verify required classes
# ------------------------------------------------------------

required_classes = {
    "glaucoma",
    "normal"
}


if not required_classes.issubset(
    class_indices.keys()
):

    raise RuntimeError(
        "\nThe class_indices.json does not contain "
        "both Glaucoma and Normal.\n\n"
        f"Found:\n{class_indices}"
    )


GLAUCOMA_INDEX = int(
    class_indices["glaucoma"]
)


NORMAL_INDEX = int(
    class_indices["normal"]
)


print(
    "Glaucoma index:",
    GLAUCOMA_INDEX
)


print(
    "Normal index:",
    NORMAL_INDEX
)


# ============================================================
# BUILD EXACT REPOSITORY ARCHITECTURE
# ============================================================
#
# Repository architecture:
#
# NASNetMobile
#      ↓
# GlobalAveragePooling2D
#      ↓
# Dropout(0.3)
#      ↓
# Dense(256, relu)
#      ↓
# Dense(4, softmax)
#
# We DO NOT call load_model().
# ============================================================

print()
print("=" * 80)
print("BUILDING MODEL ARCHITECTURE")
print("=" * 80)


keras.backend.clear_session()


base_model = keras.applications.NASNetMobile(
    weights=None,
    include_top=False,
    input_shape=(
        IMAGE_SIZE,
        IMAGE_SIZE,
        3
    )
)


x = base_model.output


x = keras.layers.GlobalAveragePooling2D()(
    x
)


x = keras.layers.Dropout(
    0.3
)(
    x
)


x = keras.layers.Dense(
    256,
    activation="relu"
)(
    x
)


output = keras.layers.Dense(
    4,
    activation="softmax"
)(
    x
)


model = keras.Model(
    inputs=base_model.input,
    outputs=output
)


print()
print(
    "Input shape :",
    model.input_shape
)


print(
    "Output shape:",
    model.output_shape
)


# ============================================================
# VERIFY MODEL STRUCTURE
# ============================================================

if tuple(model.input_shape) != (
    None,
    224,
    224,
    3
):

    raise RuntimeError(
        "\nUnexpected model input shape:\n"
        f"{model.input_shape}"
    )


if tuple(model.output_shape) != (
    None,
    4
):

    raise RuntimeError(
        "\nUnexpected model output shape:\n"
        f"{model.output_shape}"
    )


# ============================================================
# KERAS-3 WEIGHT LOADING
# ============================================================
#
# Your .keras file contains:
#
#   metadata.json
#   config.json
#   model.weights.h5
#
# Keras 3 stores weights under:
#
#   layers/<generated-layer-name>/vars/<index>
#
# TensorFlow 2.13 cannot directly load that HDF5 structure
# with model.load_weights().
#
# Therefore we map weights manually.
# ============================================================

print()
print("=" * 80)
print("READING TRAINED WEIGHTS FROM .KERAS CHECKPOINT")
print("=" * 80)


def natural_key(name):

    parts = []

    for part in name.split("_"):

        if part.isdigit():

            parts.append(
                int(part)
            )

        else:

            parts.append(
                part
            )

    return parts


# ------------------------------------------------------------
# Mapping from Keras layer class -> H5 group prefix
# ------------------------------------------------------------

CLASS_TO_PREFIX = {

    "Conv2D":
        "conv2d",

    "BatchNormalization":
        "batch_normalization",

    "SeparableConv2D":
        "separable_conv2d",

    "Dense":
        "dense",

}


# ------------------------------------------------------------
# Extract model.weights.h5 from .keras archive
# ------------------------------------------------------------

temporary_directory = tempfile.mkdtemp(
    prefix="glaucosense_weights_"
)


weights_file = (
    Path(temporary_directory)
    /
    "model.weights.h5"
)


try:

    print()
    print("Opening .keras archive...")


    with zipfile.ZipFile(
        model_path,
        "r"
    ) as archive:


        archive_names = archive.namelist()


        if (
            "model.weights.h5"
            not in archive_names
        ):

            raise RuntimeError(
                "\nmodel.weights.h5 was not found "
                "inside the .keras file."
            )


        with archive.open(
            "model.weights.h5",
            "r"
        ) as source:

            with open(
                weights_file,
                "wb"
            ) as destination:

                shutil.copyfileobj(
                    source,
                    destination
                )


    print(
        "model.weights.h5 extracted."
    )


    # ========================================================
    # OPEN H5 WEIGHTS
    # ========================================================

    with h5py.File(
        weights_file,
        "r"
    ) as h5_file:


        if "layers" not in h5_file:

            raise RuntimeError(
                "\nInvalid Keras 3 weights file:\n"
                "missing 'layers' group."
            )


        h5_layers = h5_file["layers"]


        # ----------------------------------------------------
        # Create naturally ordered H5 groups
        # ----------------------------------------------------

        grouped_weights = {}


        for class_prefix in CLASS_TO_PREFIX.values():


            matching_names = [

                name

                for name in h5_layers.keys()

                if (
                    name == class_prefix
                    or
                    name.startswith(
                        class_prefix + "_"
                    )
                )
            ]


            matching_names.sort(
                key=natural_key
            )


            grouped_weights[
                class_prefix
            ] = matching_names


        # ----------------------------------------------------
        # Print counts
        # ----------------------------------------------------

        print()
        print(
            "Checkpoint weight groups:"
        )


        for class_prefix, names in (
            grouped_weights.items()
        ):

            print(
                f"  {class_prefix:<22}"
                f"{len(names)}"
            )


        # ----------------------------------------------------
        # Expected counts
        # ----------------------------------------------------

        model_counts = {

            "conv2d":
                sum(
                    1
                    for layer in model.layers
                    if layer.__class__.__name__
                    == "Conv2D"
                ),

            "batch_normalization":
                sum(
                    1
                    for layer in model.layers
                    if layer.__class__.__name__
                    == "BatchNormalization"
                ),

            "separable_conv2d":
                sum(
                    1
                    for layer in model.layers
                    if layer.__class__.__name__
                    == "SeparableConv2D"
                ),

            "dense":
                sum(
                    1
                    for layer in model.layers
                    if layer.__class__.__name__
                    == "Dense"
                ),

        }


        print()
        print(
            "Reconstructed model weight layers:"
        )


        for prefix, count in (
            model_counts.items()
        ):

            print(
                f"  {prefix:<22}"
                f"{count}"
            )


        # ----------------------------------------------------
        # Verify counts BEFORE loading anything
        # ----------------------------------------------------

        for prefix in model_counts:

            if (
                model_counts[prefix]
                != len(
                    grouped_weights[prefix]
                )
            ):

                raise RuntimeError(
                    "\nWeight-group count mismatch.\n\n"
                    f"Layer type: {prefix}\n"
                    f"Model      : "
                    f"{model_counts[prefix]}\n"
                    f"Checkpoint : "
                    f"{len(grouped_weights[prefix])}"
                )


        # ----------------------------------------------------
        # Track which group is next for each layer type
        # ----------------------------------------------------

        group_positions = {

            prefix: 0

            for prefix
            in CLASS_TO_PREFIX.values()

        }


        total_variables = 0
        loaded_variables = 0


        # ====================================================
        # FIRST PASS:
        # VALIDATE ALL WEIGHT SHAPES
        # ====================================================

        print()
        print("=" * 80)
        print("VALIDATING CHECKPOINT WEIGHTS")
        print("=" * 80)


        assignments = []


        for layer in model.layers:


            class_name = (
                layer.__class__.__name__
            )


            if class_name not in CLASS_TO_PREFIX:

                continue


            if not layer.weights:

                continue


            prefix = CLASS_TO_PREFIX[
                class_name
            ]


            position = group_positions[
                prefix
            ]


            group_name = (
                grouped_weights[
                    prefix
                ][position]
            )


            group_positions[
                prefix
            ] += 1


            group = h5_layers[
                group_name
            ]


            if "vars" not in group:

                raise RuntimeError(
                    "\nMissing vars group:\n"
                    f"{group_name}"
                )


            vars_group = group[
                "vars"
            ]


            layer_arrays = []


            for variable_index in range(
                len(layer.weights)
            ):


                key = str(
                    variable_index
                )


                if key not in vars_group:

                    raise RuntimeError(
                        "\nMissing variable.\n\n"
                        f"Layer : {layer.name}\n"
                        f"Group : {group_name}\n"
                        f"Variable index: "
                        f"{variable_index}"
                    )


                array = np.asarray(
                    vars_group[key]
                )


                expected_shape = tuple(
                    layer.weights[
                        variable_index
                    ].shape
                )


                actual_shape = tuple(
                    array.shape
                )


                total_variables += 1


                if (
                    expected_shape
                    != actual_shape
                ):

                    raise RuntimeError(
                        "\nWEIGHT SHAPE MISMATCH\n\n"
                        f"Layer : {layer.name}\n"
                        f"Class : {class_name}\n"
                        f"Group : {group_name}\n"
                        f"Variable: {variable_index}\n\n"
                        f"Expected: {expected_shape}\n"
                        f"Found   : {actual_shape}"
                    )


                layer_arrays.append(
                    array
                )


            assignments.append(
                (
                    layer,
                    group_name,
                    layer_arrays
                )
            )


        print()
        print(
            f"Validated "
            f"{total_variables} "
            f"model variables."
        )


        # ====================================================
        # SECOND PASS:
        # ACTUALLY ASSIGN WEIGHTS
        # ====================================================

        print()
        print("=" * 80)
        print("ASSIGNING TRAINED WEIGHTS")
        print("=" * 80)


        for (
            layer,
            group_name,
            layer_arrays
        ) in assignments:


            layer.set_weights(
                layer_arrays
            )


            loaded_variables += len(
                layer_arrays
            )


        print()
        print(
            f"Loaded "
            f"{loaded_variables} "
            f"variables successfully."
        )


finally:

    # ========================================================
    # CLOSE ALL FILES FIRST
    # THEN REMOVE TEMP DIRECTORY
    # ========================================================

    shutil.rmtree(
        temporary_directory,
        ignore_errors=True
    )


# ============================================================
# MODEL READY
# ============================================================

print()
print("=" * 80)
print("TRAINED MODEL READY")
print("=" * 80)


# ============================================================
# GET IMAGE LIST
# ============================================================

def get_images(source):


    if source.is_file():


        if (
            source.suffix.lower()
            not in SUPPORTED_EXTENSIONS
        ):

            raise ValueError(
                f"\nUnsupported image format:\n"
                f"{source}"
            )


        return [source]


    if source.is_dir():


        images = sorted(

            [

                p

                for p in source.iterdir()

                if (
                    p.is_file()
                    and
                    p.suffix.lower()
                    in SUPPORTED_EXTENSIONS
                )

            ]

        )


        if not images:

            raise FileNotFoundError(
                "\nNo supported images found:\n"
                f"{source}"
            )


        return images


    raise FileNotFoundError(
        f"\nInvalid image source:\n"
        f"{source}"
    )


images = get_images(
    image_source
)


print()
print(
    f"Images found: "
    f"{len(images)}"
)


# ============================================================
# REMOVE BLACK BORDER
# ============================================================

def remove_black_border(
    image
):


    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )


    _, threshold = cv2.threshold(
        gray,
        10,
        255,
        cv2.THRESH_BINARY
    )


    coords = np.column_stack(
        np.where(
            threshold > 0
        )
    )


    if coords.size == 0:

        return image


    x, y, w, h = cv2.boundingRect(
        coords
    )


    cropped = image[
        y:y+h,
        x:x+w
    ]


    if cropped.size == 0:

        return image


    return cropped


# ============================================================
# PREPROCESS IMAGE
# ============================================================

def preprocess_image(
    image_path
):


    image = cv2.imread(
        str(image_path),
        cv2.IMREAD_UNCHANGED
    )


    if image is None:

        raise RuntimeError(
            f"\nUnable to read image:\n"
            f"{image_path}"
        )


    original_height = (
        image.shape[0]
    )


    original_width = (
        image.shape[1]
    )


    # --------------------------------------------------------
    # GRAYSCALE
    # --------------------------------------------------------

    if image.ndim == 2:


        # Duplicate grayscale intensity
        # into three channels.
        #
        # No colorization.
        # No synthetic color.

        image = np.stack(
            [
                image,
                image,
                image
            ],
            axis=-1
        )


        image_type = (
            "GRAYSCALE"
        )


    # --------------------------------------------------------
    # SINGLE CHANNEL
    # --------------------------------------------------------

    elif (
        image.ndim == 3
        and image.shape[2] == 1
    ):


        image = np.repeat(
            image,
            3,
            axis=2
        )


        image_type = (
            "GRAYSCALE"
        )


    # --------------------------------------------------------
    # FOUR CHANNEL
    # --------------------------------------------------------

    elif (
        image.ndim == 3
        and image.shape[2] == 4
    ):


        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGRA2BGR
        )


        image_type = (
            "COLOUR"
        )


    # --------------------------------------------------------
    # THREE CHANNEL
    # --------------------------------------------------------

    elif (
        image.ndim == 3
        and image.shape[2] == 3
    ):


        image_type = (
            "COLOUR"
        )


    else:

        raise RuntimeError(
            f"\nUnsupported image shape:\n"
            f"{image.shape}"
        )


    # --------------------------------------------------------
    # DATA TYPE
    # --------------------------------------------------------

    if image.dtype == np.uint8:

        pass


    elif image.dtype == np.uint16:

        image = (
            image.astype(
                np.float32
            )
            / 257.0
        ).clip(
            0,
            255
        ).astype(
            np.uint8
        )


    else:

        raise RuntimeError(
            f"\nUnsupported image dtype:\n"
            f"{image.dtype}"
        )


    # --------------------------------------------------------
    # BLACK BORDER
    # --------------------------------------------------------

    image = remove_black_border(
        image
    )


    # --------------------------------------------------------
    # RESIZE
    # --------------------------------------------------------

    image = cv2.resize(
        image,
        (
            IMAGE_SIZE,
            IMAGE_SIZE
        ),
        interpolation=cv2.INTER_AREA
    )


    # --------------------------------------------------------
    # RGB
    # --------------------------------------------------------

    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )


    # --------------------------------------------------------
    # NORMALIZATION
    # --------------------------------------------------------

    image = (
        image.astype(
            np.float32
        )
        / 255.0
    )


    # --------------------------------------------------------
    # BATCH
    # --------------------------------------------------------

    image = np.expand_dims(
        image,
        axis=0
    )


    print()
    print(
        f"Original image : "
        f"{original_width} x "
        f"{original_height}"
    )


    print(
        f"Image type     : "
        f"{image_type}"
    )


    print(
        f"Model input    : "
        f"{image.shape}"
    )


    return image


# ============================================================
# INFERENCE
# ============================================================

print()
print("=" * 80)
print("RUNNING INFERENCE")
print("=" * 80)


for number, image_path in enumerate(
    images,
    start=1
):


    print()
    print("-" * 80)
    print(
        f"IMAGE {number}/{len(images)}"
    )
    print("-" * 80)


    print(
        f"File: {image_path.name}"
    )


    # --------------------------------------------------------
    # PREPROCESS
    # --------------------------------------------------------

    image = preprocess_image(
        image_path
    )


    # --------------------------------------------------------
    # PREDICT
    # --------------------------------------------------------
    #
    # Direct model call.
    # No prediction progress bar.
    # --------------------------------------------------------

    prediction = (
        model(
            image,
            training=False
        )
        .numpy()[0]
    )


    # --------------------------------------------------------
    # VALIDATE NUMBERS
    # --------------------------------------------------------

    if prediction.shape != (4,):

        raise RuntimeError(
            "\nUnexpected prediction shape:\n"
            f"{prediction.shape}"
        )


    if not np.all(
        np.isfinite(
            prediction
        )
    ):

        raise RuntimeError(
            "\nModel produced invalid numerical output."
        )


    # --------------------------------------------------------
    # NORMALIZE
    # --------------------------------------------------------

    probability_sum = float(
        np.sum(
            prediction
        )
    )


    if probability_sum <= 0:

        raise RuntimeError(
            "\nInvalid probability output."
        )


    prediction = (
        prediction
        /
        probability_sum
    )


    # --------------------------------------------------------
    # TARGET PROBABILITIES
    # --------------------------------------------------------

    glaucoma_probability = float(
        prediction[
            GLAUCOMA_INDEX
        ]
    )


    normal_probability = float(
        prediction[
            NORMAL_INDEX
        ]
    )


    # --------------------------------------------------------
    # TOP CLASS
    # --------------------------------------------------------

    top_index = int(
        np.argmax(
            prediction
        )
    )


    # --------------------------------------------------------
    # STRICT TARGET DECISION
    # --------------------------------------------------------

    if (
        top_index
        ==
        GLAUCOMA_INDEX
    ):

        result = (
            "Glaucoma"
        )


    elif (
        top_index
        ==
        NORMAL_INDEX
    ):

        result = (
            "Normal"
        )


    else:

        result = (
            "NO DECISION - "
            "OUTSIDE TARGET SCOPE"
        )


    # ========================================================
    # RESULT
    # ========================================================

    print()
    print("=" * 80)
    print("RESULT")
    print("=" * 80)


    print(
        f"Glaucoma probability : "
        f"{glaucoma_probability:.2%}"
    )


    print(
        f"Normal probability   : "
        f"{normal_probability:.2%}"
    )


    print()
    print(
        "FINAL RESULT         :",
        result
    )


    print("=" * 80)


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 80)
print("INFERENCE COMPLETE")
print("=" * 80)