import os
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_PATH = (
    r"D:\Anurag BPCL WORK\Glaucosense\TASK NO. PRJ-002111"
    r"\ESRGAN_Model_Result\1_OD_1.tif"
)

CHECKPOINT_PATH = (
    r"D:\Anurag BPCL WORK\Glaucosense\TASK NO. PRJ-002111"
    r"\EyeeSEE\checkpoints\checkpoint_epoch_057.pt"
)

OUTPUT_DIR = (
    r"D:\Anurag BPCL WORK\Glaucosense\TASK NO. PRJ-002111"
    r"\Glaucoma Results"
)

IMAGE_SIZE = 512


# ============================================================
# CHECKPOINT-INDICATED CLASS STRUCTURE
# ============================================================
#
# The checkpoint has 3 output channels.
#
# At this stage we use:
#
# 0 = Background
# 1 = Optic Disc
# 2 = Optic Cup
#
# This class interpretation must still be independently
# verified against the original training information.
#
# We do NOT alter the prediction based on this assumption.
# ============================================================

BACKGROUND_CLASS = 0
DISC_CLASS = 1
CUP_CLASS = 2


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# EXACT CHECKPOINT-STYLE DOUBLE CONV
# ============================================================
#
# IMPORTANT:
# The checkpoint keys show:
#
# inc.conv.0
# inc.conv.1
# inc.conv.3
# inc.conv.4
#
# Therefore the container must be named "conv".
# ============================================================

class DoubleConv(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels
    ):

        super().__init__()

        self.conv = nn.Sequential(

            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            ),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),

            nn.BatchNorm2d(
                out_channels
            ),

            nn.ReLU(
                inplace=True
            )
        )

    def forward(self, x):

        return self.conv(x)


# ============================================================
# EXACT CHECKPOINT-STYLE DOWN
# ============================================================
#
# Checkpoint keys:
#
# down1.pool_conv.1.conv.0
#
# Therefore:
#
# self.pool_conv
#     ├── MaxPool2d
#     └── DoubleConv
# ============================================================

class Down(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels
    ):

        super().__init__()

        self.pool_conv = nn.Sequential(

            nn.MaxPool2d(
                kernel_size=2
            ),

            DoubleConv(
                in_channels,
                out_channels
            )
        )

    def forward(self, x):

        return self.pool_conv(x)


# ============================================================
# EXACT CHECKPOINT-STYLE UP
# ============================================================

class Up(nn.Module):

    def __init__(
        self,
        in_channels,
        out_channels
    ):

        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_channels,
            in_channels // 2,
            kernel_size=2,
            stride=2
        )

        self.conv = DoubleConv(
            in_channels,
            out_channels
        )

    def forward(
        self,
        x1,
        x2
    ):

        x1 = self.up(x1)

        # ----------------------------------------------------
        # Padding is ONLY to make tensor dimensions compatible
        # for concatenation.
        #
        # It does not modify the final predicted mask.
        # ----------------------------------------------------

        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)

        x1 = F.pad(
            x1,
            [
                diff_x // 2,
                diff_x - diff_x // 2,
                diff_y // 2,
                diff_y - diff_y // 2
            ]
        )

        x = torch.cat(
            [x2, x1],
            dim=1
        )

        return self.conv(x)


# ============================================================
# U-NET
# ============================================================

class UNet(nn.Module):

    def __init__(self):

        super().__init__()

        # ----------------------------------------------------
        # Encoder
        # ----------------------------------------------------

        self.inc = DoubleConv(
            1,
            64
        )

        self.down1 = Down(
            64,
            128
        )

        self.down2 = Down(
            128,
            256
        )

        self.down3 = Down(
            256,
            512
        )

        self.down4 = Down(
            512,
            1024
        )

        # ----------------------------------------------------
        # Decoder
        # ----------------------------------------------------

        self.up1 = Up(
            1024,
            512
        )

        self.up2 = Up(
            512,
            256
        )

        self.up3 = Up(
            256,
            128
        )

        self.up4 = Up(
            128,
            64
        )

        # ----------------------------------------------------
        # Three output classes
        # ----------------------------------------------------

        self.outc = nn.Conv2d(
            64,
            3,
            kernel_size=1
        )

    def forward(self, x):

        x1 = self.inc(x)

        x2 = self.down1(x1)

        x3 = self.down2(x2)

        x4 = self.down3(x3)

        x5 = self.down4(x4)

        x = self.up1(
            x5,
            x4
        )

        x = self.up2(
            x,
            x3
        )

        x = self.up3(
            x,
            x2
        )

        x = self.up4(
            x,
            x1
        )

        logits = self.outc(x)

        return logits


# ============================================================
# LOAD CHECKPOINT
# ============================================================

def load_model():

    print("=" * 70)
    print("EYEESEE - GENUINE RAW INFERENCE")
    print("=" * 70)

    print(
        "Device:",
        DEVICE
    )

    if DEVICE.type == "cuda":

        print(
            "GPU:",
            torch.cuda.get_device_name(0)
        )

    print("\nLoading checkpoint:")
    print(CHECKPOINT_PATH)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE
    )

    print("\nCheckpoint information:")

    if isinstance(checkpoint, dict):

        print(
            "Checkpoint keys:",
            list(checkpoint.keys())
        )

        if "epoch" in checkpoint:

            print(
                "Epoch:",
                checkpoint["epoch"]
            )

        if "best_val_loss" in checkpoint:

            print(
                "Best validation loss:",
                checkpoint["best_val_loss"]
            )

        if "val_metrics" in checkpoint:

            print(
                "Validation metrics:",
                checkpoint["val_metrics"]
            )

        if "model_state_dict" not in checkpoint:

            raise RuntimeError(
                "Checkpoint does not contain "
                "'model_state_dict'."
            )

        state_dict = checkpoint[
            "model_state_dict"
        ]

    else:

        raise RuntimeError(
            "Unexpected checkpoint format."
        )

    # --------------------------------------------------------
    # Build EXACT checkpoint-style architecture.
    # --------------------------------------------------------

    model = UNet().to(DEVICE)

    print(
        "\nLoading model weights with strict=True..."
    )

    model.load_state_dict(
        state_dict,
        strict=True
    )

    model.eval()

    print(
        "\nSUCCESS:"
    )

    print(
        "Checkpoint loaded with strict=True."
    )

    print(
        "No checkpoint weights were skipped."
    )

    print(
        "No checkpoint weights were randomly initialized."
    )

    return model, checkpoint


# ============================================================
# LOAD IMAGE
# ============================================================

def load_image():

    print(
        "\nLoading image:"
    )

    print(
        IMAGE_PATH
    )

    image = cv2.imread(
        IMAGE_PATH,
        cv2.IMREAD_UNCHANGED
    )

    if image is None:

        raise FileNotFoundError(
            f"\nCould not read:\n{IMAGE_PATH}"
        )

    print(
        "Original image shape:",
        image.shape
    )

    print(
        "Original image dtype:",
        image.dtype
    )

    return image


# ============================================================
# PREPROCESS
# ============================================================

def preprocess_image(image):

    # --------------------------------------------------------
    # The checkpoint architecture requires ONE input channel.
    # --------------------------------------------------------

    if image.ndim == 3:

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

    elif image.ndim == 2:

        gray = image.copy()

    else:

        raise ValueError(
            f"Unsupported image shape: {image.shape}"
        )

    # --------------------------------------------------------
    # Resize to model input resolution.
    # --------------------------------------------------------

    resized = cv2.resize(
        gray,
        (
            IMAGE_SIZE,
            IMAGE_SIZE
        ),
        interpolation=cv2.INTER_AREA
    )

    # --------------------------------------------------------
    # Convert to float.
    # --------------------------------------------------------

    image_float = resized.astype(
        np.float32
    )

    # --------------------------------------------------------
    # Current preprocessing assumption:
    # 0-255 → 0-1
    #
    # IMPORTANT:
    # This has NOT yet been independently confirmed from
    # the original training code.
    # --------------------------------------------------------

    image_float /= 255.0

    # --------------------------------------------------------
    # H,W
    # →
    # 1,H,W
    # →
    # 1,1,H,W
    # --------------------------------------------------------

    tensor = torch.from_numpy(
        image_float
    )

    tensor = tensor.unsqueeze(0)

    tensor = tensor.unsqueeze(0)

    tensor = tensor.to(DEVICE)

    print(
        "\nModel input tensor:"
    )

    print(
        "Shape:",
        tuple(tensor.shape)
    )

    print(
        "Min:",
        tensor.min().item()
    )

    print(
        "Max:",
        tensor.max().item()
    )

    return gray, resized, tensor


# ============================================================
# RAW INFERENCE
# ============================================================

@torch.no_grad()
def run_model(
    model,
    tensor
):

    logits = model(
        tensor
    )

    probabilities = torch.softmax(
        logits,
        dim=1
    )

    prediction = torch.argmax(
        probabilities,
        dim=1
    )

    print(
        "\nModel output:"
    )

    print(
        "Logits shape:",
        tuple(logits.shape)
    )

    print(
        "Probabilities shape:",
        tuple(probabilities.shape)
    )

    print(
        "Prediction shape:",
        tuple(prediction.shape)
    )

    return (
        logits,
        probabilities,
        prediction
    )


# ============================================================
# EXTRACT RAW MASKS
# ============================================================

def extract_raw_masks(
    prediction
):

    prediction = (
        prediction[0]
        .detach()
        .cpu()
        .numpy()
    )

    background_mask = (
        prediction == BACKGROUND_CLASS
    )

    disc_mask = (
        prediction == DISC_CLASS
    )

    cup_mask = (
        prediction == CUP_CLASS
    )

    return (
        prediction,
        background_mask,
        disc_mask,
        cup_mask
    )


# ============================================================
# RAW STATISTICS
# ============================================================

def print_statistics(
    prediction,
    probabilities
):

    background_pixels = int(
        np.sum(
            prediction == BACKGROUND_CLASS
        )
    )

    disc_pixels = int(
        np.sum(
            prediction == DISC_CLASS
        )
    )

    cup_pixels = int(
        np.sum(
            prediction == CUP_CLASS
        )
    )

    print(
        "\nRaw predicted pixels:"
    )

    print(
        "Background pixels:",
        background_pixels
    )

    print(
        "Optic Disc pixels:",
        disc_pixels
    )

    print(
        "Optic Cup pixels:",
        cup_pixels
    )

    mean_probabilities = (
        probabilities[0]
        .mean(
            dim=(1, 2)
        )
        .detach()
        .cpu()
        .numpy()
    )

    print(
        "\nMean probabilities:"
    )

    for i, probability in enumerate(
        mean_probabilities
    ):

        print(
            f"Class {i}: "
            f"{probability:.6f}"
        )

    return (
        background_pixels,
        disc_pixels,
        cup_pixels,
        mean_probabilities
    )


# ============================================================
# RAW VERTICAL DIAMETER
# ============================================================

def get_vertical_diameter(
    mask
):

    ys, xs = np.where(
        mask
    )

    if len(ys) == 0:

        return None

    return int(
        ys.max() - ys.min() + 1
    )


# ============================================================
# RAW VCDR
# ============================================================

def calculate_raw_vcdr(
    disc_mask,
    cup_mask
):

    disc_diameter = (
        get_vertical_diameter(
            disc_mask
        )
    )

    cup_diameter = (
        get_vertical_diameter(
            cup_mask
        )
    )

    print(
        "\nRaw mask geometry:"
    )

    print(
        "Disc vertical diameter:",
        disc_diameter
    )

    print(
        "Cup vertical diameter:",
        cup_diameter
    )

    # --------------------------------------------------------
    # Missing disc
    # --------------------------------------------------------

    if disc_diameter is None:

        return {
            "valid": False,
            "vcdr": None,
            "reason": "NO_DISC_PREDICTED",
            "disc_vertical_diameter": None,
            "cup_vertical_diameter": cup_diameter
        }

    # --------------------------------------------------------
    # Missing cup
    # --------------------------------------------------------

    if cup_diameter is None:

        return {
            "valid": False,
            "vcdr": None,
            "reason": "NO_CUP_PREDICTED",
            "disc_vertical_diameter": disc_diameter,
            "cup_vertical_diameter": None
        }

    # --------------------------------------------------------
    # Impossible raw geometry.
    #
    # DO NOT repair.
    # --------------------------------------------------------

    if cup_diameter > disc_diameter:

        return {
            "valid": False,
            "vcdr": None,
            "reason": (
                "RAW_CUP_LARGER_THAN_RAW_DISC"
            ),
            "disc_vertical_diameter": disc_diameter,
            "cup_vertical_diameter": cup_diameter
        }

    # --------------------------------------------------------
    # Equal diameter is treated as invalid.
    # --------------------------------------------------------

    if cup_diameter == disc_diameter:

        return {
            "valid": False,
            "vcdr": None,
            "reason": (
                "RAW_CUP_DIAMETER_EQUALS_RAW_DISC"
            ),
            "disc_vertical_diameter": disc_diameter,
            "cup_vertical_diameter": cup_diameter
        }

    # --------------------------------------------------------
    # Genuine vCDR.
    # --------------------------------------------------------

    vcdr = (
        cup_diameter /
        disc_diameter
    )

    return {
        "valid": True,
        "vcdr": float(vcdr),
        "reason": "VALID_RAW_GEOMETRY",
        "disc_vertical_diameter": disc_diameter,
        "cup_vertical_diameter": cup_diameter
    }


# ============================================================
# SAVE RAW MASKS
# ============================================================

def save_raw_outputs(
    resized,
    prediction,
    disc_mask,
    cup_mask
):

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    base_name = os.path.splitext(
        os.path.basename(
            IMAGE_PATH
        )
    )[0]

    # --------------------------------------------------------
    # 1. Raw class mask
    #
    # Pixel values:
    #
    # 0 = background
    # 1 = disc
    # 2 = cup
    #
    # These values come directly from argmax.
    # --------------------------------------------------------

    raw_mask_path = os.path.join(
        OUTPUT_DIR,
        f"{base_name}_RAW_MASK.png"
    )

    cv2.imwrite(
        raw_mask_path,
        prediction.astype(
            np.uint8
        )
    )

    # --------------------------------------------------------
    # 2. Raw disc mask
    # --------------------------------------------------------

    disc_mask_path = os.path.join(
        OUTPUT_DIR,
        f"{base_name}_RAW_DISC_MASK.png"
    )

    cv2.imwrite(
        disc_mask_path,
        (
            disc_mask.astype(
                np.uint8
            ) * 255
        )
    )

    # --------------------------------------------------------
    # 3. Raw cup mask
    # --------------------------------------------------------

    cup_mask_path = os.path.join(
        OUTPUT_DIR,
        f"{base_name}_RAW_CUP_MASK.png"
    )

    cv2.imwrite(
        cup_mask_path,
        (
            cup_mask.astype(
                np.uint8
            ) * 255
        )
    )

    # --------------------------------------------------------
    # 4. Visualization overlay
    #
    # IMPORTANT:
    # The overlay does NOT modify the masks.
    # It is only a visualization of the raw prediction.
    # --------------------------------------------------------

    overlay = cv2.cvtColor(
        resized,
        cv2.COLOR_GRAY2BGR
    )

    # Disc shown in blue.
    overlay[disc_mask] = (
        255,
        0,
        0
    )

    # Cup shown in green.
    overlay[cup_mask] = (
        0,
        255,
        0
    )

    overlay_path = os.path.join(
        OUTPUT_DIR,
        f"{base_name}_RAW_OVERLAY.png"
    )

    cv2.imwrite(
        overlay_path,
        overlay
    )

    print(
        "\nSaved raw model outputs:"
    )

    print(
        raw_mask_path
    )

    print(
        disc_mask_path
    )

    print(
        cup_mask_path
    )

    print(
        overlay_path
    )


# ============================================================
# SAVE JSON
# ============================================================

def save_json(
    checkpoint,
    original,
    class_counts,
    mean_probabilities,
    vcdr_result
):

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    base_name = os.path.splitext(
        os.path.basename(
            IMAGE_PATH
        )
    )[0]

    json_path = os.path.join(
        OUTPUT_DIR,
        f"{base_name}_genuine_result.json"
    )

    result = {

        "image_path":
            IMAGE_PATH,

        "checkpoint_path":
            CHECKPOINT_PATH,

        "device":
            str(DEVICE),

        "input_size":
            [
                IMAGE_SIZE,
                IMAGE_SIZE
            ],

        "original_image_shape":
            list(
                original.shape
            ),

        "class_mapping":
            {
                "0": "background",
                "1": "optic_disc",
                "2": "optic_cup"
            },

        "raw_predicted_pixel_counts":
            {
                "background":
                    int(
                        class_counts[0]
                    ),

                "optic_disc":
                    int(
                        class_counts[1]
                    ),

                "optic_cup":
                    int(
                        class_counts[2]
                    )
            },

        "mean_class_probabilities":
            {
                f"class_{i}":
                    float(
                        probability
                    )

                for i, probability
                in enumerate(
                    mean_probabilities
                )
            },

        "vcdr_result":
            vcdr_result,

        "synthetic_processing":
            False,

        "mask_repair":
            False,

        "morphological_processing":
            False,

        "geometric_correction":
            False,

        "vcdr_clamping":
            False,

        "description":
            (
                "Segmentation and vCDR are derived "
                "from the raw model prediction. "
                "No synthetic mask generation or "
                "mask repair was performed."
            )
    }

    if isinstance(
        checkpoint,
        dict
    ):

        if "epoch" in checkpoint:

            result[
                "checkpoint_epoch"
            ] = checkpoint[
                "epoch"
            ]

        if "best_val_loss" in checkpoint:

            result[
                "checkpoint_best_val_loss"
            ] = checkpoint[
                "best_val_loss"
            ]

        if "val_metrics" in checkpoint:

            result[
                "checkpoint_val_metrics"
            ] = checkpoint[
                "val_metrics"
            ]

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            result,
            f,
            indent=4
        )

    print(
        "\nJSON result saved:"
    )

    print(
        json_path
    )


# ============================================================
# MAIN
# ============================================================

def main():

    # --------------------------------------------------------
    # 1. Load checkpoint
    # --------------------------------------------------------

    model, checkpoint = load_model()

    # --------------------------------------------------------
    # 2. Load original image
    # --------------------------------------------------------

    original = load_image()

    # --------------------------------------------------------
    # 3. Preprocess
    # --------------------------------------------------------

    (
        gray,
        resized,
        tensor
    ) = preprocess_image(
        original
    )

    # --------------------------------------------------------
    # 4. Genuine model inference
    # --------------------------------------------------------

    (
        logits,
        probabilities,
        prediction
    ) = run_model(
        model,
        tensor
    )

    # --------------------------------------------------------
    # 5. Extract raw masks
    # --------------------------------------------------------

    (
        prediction_np,
        background_mask,
        disc_mask,
        cup_mask
    ) = extract_raw_masks(
        prediction
    )

    # --------------------------------------------------------
    # 6. Statistics
    # --------------------------------------------------------

    (
        background_pixels,
        disc_pixels,
        cup_pixels,
        mean_probabilities
    ) = print_statistics(
        prediction_np,
        probabilities
    )

    class_counts = {
        0: background_pixels,
        1: disc_pixels,
        2: cup_pixels
    }

    # --------------------------------------------------------
    # 7. Raw vCDR
    # --------------------------------------------------------

    vcdr_result = calculate_raw_vcdr(
        disc_mask,
        cup_mask
    )

    # --------------------------------------------------------
    # 8. Final result
    # --------------------------------------------------------

    print(
        "\n"
        + "=" * 70
    )

    print(
        "FINAL RESULT"
    )

    print(
        "=" * 70
    )

    if vcdr_result["valid"]:

        print(
            "Raw vCDR:",
            f"{vcdr_result['vcdr']:.4f}"
        )

        print(
            "Status: VALID RAW GEOMETRY"
        )

    else:

        print(
            "Raw vCDR: NOT CALCULATED"
        )

        print(
            "Status: MANUAL REVIEW"
        )

        print(
            "Reason:",
            vcdr_result["reason"]
        )

    # --------------------------------------------------------
    # 9. Save raw outputs
    # --------------------------------------------------------

    save_raw_outputs(
        resized,
        prediction_np,
        disc_mask,
        cup_mask
    )

    # --------------------------------------------------------
    # 10. Save JSON
    # --------------------------------------------------------

    save_json(
        checkpoint,
        original,
        class_counts,
        mean_probabilities,
        vcdr_result
    )

    print(
        "\n"
        + "=" * 70
    )

    print(
        "INFERENCE COMPLETE"
    )

    print(
        "=" * 70
    )


if __name__ == "__main__":

    main()