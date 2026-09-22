"""
GLAUCOMA INFERENCE SCRIPT
-------------------------
Purpose:
    1. Load the pretrained golden-panther glaucoma model (.h5).
    2. Read full fundus images (JPG/JPEG/PNG/BMP/TIF/TIFF).
    3. Convert every image to RGB because the model expects 3 channels.
    4. Automatically estimate the optic-disc/optic-nerve-head center.
    5. Create a square crop around that center.
    6. Open an interactive review window BEFORE inference.
       - Drag the crop on the original image.
       - Mouse wheel / +/- changes crop size.
       - Arrow keys make small position adjustments.
       - R resets to the automatic crop.
       - ENTER accepts the crop.
       - ESC skips the image.
    7. After ENTER, the terminal asks YES/NO before inference.
    8. Run the exact model-style preprocessing: RGB -> 100x100 -> /255.
    9. Print Glaucoma / No Glaucoma and the model's score.

Important:
    - No ESRGAN is performed here.
    - No synthetic/fake data is created.
    - Images are processed one at a time; crops are never mixed between images.
    - The original full-resolution image is kept unchanged in memory.
"""

from __future__ import annotations

import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import math
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf
from PIL import Image, ImageOps


# ============================================================
# 1. MANUALLY SET THESE TWO PATHS
# ============================================================

MODEL_PATH = r"D:\Anurag BPCL WORK\Glaucosense\TASK NO. PRJ-002111\glaucoma-detector-master\my_model2.h5"

# Can be:
#   - one image file, OR
#   - a folder containing multiple fundus images
IMAGE_PATH = r"D:\Anurag BPCL WORK\Glaucosense\TASK NO. PRJ-002111\glaucoma-detector-master\ESRGAN_Result\1_OD_1_green.tif"

# Optional:
# SAVE_ACCEPTED_CROPS = True will save the final reviewed 100x100 
# model-input image next to each source image in "_reviewed_crops".
# Keep False if you do not want any files written.
SAVE_ACCEPTED_CROPS = False


# ============================================================
# 2. MODEL INPUT / CROPPING SETTINGS
# ============================================================

MODEL_INPUT_SIZE = (100, 100)

# Default crop size = this fraction of the detected retinal-field diameter.
# The interactive window lets you change it before inference.
AUTO_CROP_FRACTION = 0.18

MIN_CROP_PIXELS = 220
MAX_CROP_FRACTION_OF_IMAGE = 0.45

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}


# ============================================================
# 3. IMAGE LOADING
# ============================================================

def load_rgb_image(path: Path) -> Image.Image:
    """
    Load any supported image and return a true RGB PIL image.
    """
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)

        # Handles:
        #   L
        #   grayscale + alpha
        #   RGB
        #   RGBA
        #   palette
        #   TIFF
        # etc.
        #
        # The classifier expects 3 channels.
        return im.convert("RGB").copy()


def pil_to_rgb_array(image: Image.Image) -> np.ndarray:
    """
    PIL RGB -> uint8 RGB NumPy array.
    """
    return np.asarray(
        image.convert("RGB"),
        dtype=np.uint8
    )


# ============================================================
# 4. AUTOMATIC OPTIC-DISC CENTER DETECTION
# ============================================================

def _largest_retinal_component(
    gray: np.ndarray
) -> tuple[np.ndarray, float, float, float]:
    """
    Estimate the retinal field from the non-black region.

    Returns:
        mask,
        retinal_center_x,
        retinal_center_y,
        retinal_radius
    """

    h, w = gray.shape

    # Fundus images have a dark/black background around the retina.
    # This threshold tries to isolate the retinal field.
    base = (gray > 8).astype(np.uint8) * 255

    kernel_size = max(
        3,
        int(round(min(h, w) * 0.004))
    )

    if kernel_size % 2 == 0:
        kernel_size += 1

    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size)
    )

    base = cv2.morphologyEx(
        base,
        cv2.MORPH_CLOSE,
        kernel
    )

    count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        base,
        8
    )

    if count <= 1:

        mask = base

        cx = w / 2.0
        cy = h / 2.0

        radius = min(w, h) * 0.48

        return (
            mask,
            cx,
            cy,
            radius
        )

    # Largest connected component = retinal field
    largest_idx = (
        1
        + int(
            np.argmax(
                stats[1:, cv2.CC_STAT_AREA]
            )
        )
    )

    mask = (
        labels == largest_idx
    ).astype(np.uint8) * 255

    cx, cy = centroids[largest_idx]

    bw = stats[
        largest_idx,
        cv2.CC_STAT_WIDTH
    ]

    bh = stats[
        largest_idx,
        cv2.CC_STAT_HEIGHT
    ]

    radius = 0.5 * min(
        float(bw),
        float(bh)
    )

    if radius <= 0:
        radius = min(w, h) * 0.48

    return (
        mask,
        float(cx),
        float(cy),
        float(radius)
    )


def detect_optic_disc_center(
    rgb: np.ndarray
) -> tuple[tuple[float, float], float]:
    """
    Detect a bright optic-disc / optic-nerve-head region.

    The routine uses only the image itself.

    Steps:
        1. Identify retinal field.
        2. Use green channel.
        3. Find very bright regions.
        4. Generate candidate optic-disc regions.
        5. Score candidates using:
             - brightness
             - local contrast
             - circularity
             - plausible position
        6. Return the best candidate.

    Returns:
        ((center_x, center_y), retinal_diameter_estimate)
    """

    h, w = rgb.shape[:2]

    # --------------------------------------------------------
    # Downscale only for detection.
    # Final crop still comes from original full resolution.
    # --------------------------------------------------------

    max_detector_dim = 1400

    scale = min(
        1.0,
        max_detector_dim / max(h, w)
    )

    if scale < 1.0:

        small = cv2.resize(
            rgb,
            (
                max(
                    1,
                    int(round(w * scale))
                ),
                max(
                    1,
                    int(round(h * scale))
                )
            ),
            interpolation=cv2.INTER_AREA
        )

    else:

        small = rgb.copy()

    sh, sw = small.shape[:2]

    # --------------------------------------------------------
    # Retinal mask
    # --------------------------------------------------------

    gray = cv2.cvtColor(
        small,
        cv2.COLOR_RGB2GRAY
    )

    retina_mask, rcx, rcy, rr = _largest_retinal_component(
        gray
    )

    # --------------------------------------------------------
    # Green channel
    # --------------------------------------------------------

    green = small[:, :, 1]

    blur_k = max(
        9,
        int(round(min(sw, sh) * 0.008))
    )

    if blur_k % 2 == 0:
        blur_k += 1

    green_blur = cv2.GaussianBlur(
        green,
        (blur_k, blur_k),
        0
    )

    # --------------------------------------------------------
    # Retinal pixels
    # --------------------------------------------------------

    retina_values = green_blur[
        retina_mask > 0
    ]

    if retina_values.size == 0:

        return (
            (
                w / 2.0,
                h / 2.0
            ),
            min(w, h) * 0.18
        )

    # --------------------------------------------------------
    # Bright candidate regions
    # --------------------------------------------------------

    threshold = float(
        np.percentile(
            retina_values,
            99.0
        )
    )

    bright = (
        (green_blur >= threshold)
        &
        (retina_mask > 0)
    ).astype(np.uint8) * 255

    close_k = max(
        9,
        int(round(min(sw, sh) * 0.015))
    )

    if close_k % 2 == 0:
        close_k += 1

    close_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (close_k, close_k)
    )

    bright = cv2.morphologyEx(
        bright,
        cv2.MORPH_CLOSE,
        close_kernel
    )

    bright = cv2.dilate(
        bright,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (5, 5)
        ),
        iterations=1
    )

    # --------------------------------------------------------
    # Connected components
    # --------------------------------------------------------

    n, labels, stats, centroids = (
        cv2.connectedComponentsWithStats(
            bright,
            8
        )
    )

    retina_area = float(
        np.count_nonzero(retina_mask)
    )

    candidates = []

    yy, xx = np.ogrid[:sh, :sw]

    # --------------------------------------------------------
    # Evaluate candidate regions
    # --------------------------------------------------------

    for idx in range(1, n):

        area = float(
            stats[
                idx,
                cv2.CC_STAT_AREA
            ]
        )

        # Reject tiny noise.
        if area < retina_area * 0.0002:
            continue

        # Reject extremely large bright areas.
        if area > retina_area * 0.02:
            continue

        cx, cy = map(
            float,
            centroids[idx]
        )

        normalized_distance = (
            math.hypot(
                cx - rcx,
                cy - rcy
            )
            /
            max(rr, 1.0)
        )

        # Reject:
        # - center of retinal image
        # - extreme edge
        if normalized_distance < 0.05:
            continue

        if normalized_distance > 0.98:
            continue

        component_mask = (
            labels == idx
        )

        mean_brightness = float(
            green_blur[
                component_mask
            ].mean()
        )

        max_brightness = float(
            green_blur[
                component_mask
            ].max()
        )

        equivalent_radius = max(
            5.0,
            math.sqrt(
                area / math.pi
            )
        )

        # ----------------------------------------------------
        # Local contrast
        # ----------------------------------------------------

        inner = (
            (xx - cx) ** 2
            +
            (yy - cy) ** 2
        ) <= (
            equivalent_radius * 0.8
        ) ** 2

        outer = (
            (xx - cx) ** 2
            +
            (yy - cy) ** 2
        ) <= (
            equivalent_radius * 1.8
        ) ** 2

        ring = (
            outer
            &
            (~inner)
            &
            (retina_mask > 0)
        )

        if ring.any():

            ring_mean = float(
                green_blur[
                    ring
                ].mean()
            )

        else:

            ring_mean = mean_brightness

        local_contrast = max(
            0.0,
            mean_brightness - ring_mean
        )

        # ----------------------------------------------------
        # Circularity
        # ----------------------------------------------------

        contours, _ = cv2.findContours(
            (
                component_mask.astype(
                    np.uint8
                )
                * 255
            ),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:

            largest_contour = max(
                contours,
                key=cv2.contourArea
            )

            perimeter = cv2.arcLength(
                largest_contour,
                True
            )

            if perimeter > 0:

                circularity = (
                    4.0
                    * math.pi
                    * area
                    /
                    (perimeter ** 2)
                )

            else:

                circularity = 0.0

        else:

            circularity = 0.0

        # ----------------------------------------------------
        # Candidate scores
        # ----------------------------------------------------

        brightness_score = (
            mean_brightness / 255.0
        )

        contrast_score = min(
            local_contrast / 80.0,
            1.0
        )

        circularity_score = min(
            circularity,
            1.0
        )

        position_score = max(
            0.0,
            1.0
            -
            abs(
                normalized_distance - 0.55
            )
        )

        score = (
            0.55 * brightness_score
            +
            0.25 * contrast_score
            +
            0.10 * circularity_score
            +
            0.10 * position_score
        )

        candidates.append(
            (
                score,
                cx,
                cy,
                area,
                mean_brightness,
                max_brightness,
                local_contrast,
                circularity
            )
        )

    # --------------------------------------------------------
    # Select best candidate
    # --------------------------------------------------------

    if candidates:

        candidates.sort(
            reverse=True,
            key=lambda item: item[0]
        )

        (
            _,
            cx,
            cy,
            *_
        ) = candidates[0]

        full_cx = (
            cx / scale
        )

        full_cy = (
            cy / scale
        )

        retinal_diameter = (
            2.0 * rr / scale
        )

        return (
            (
                full_cx,
                full_cy
            ),
            retinal_diameter
        )

    # --------------------------------------------------------
    # FALLBACK
    # --------------------------------------------------------

    score_img = (
        green_blur.astype(
            np.float32
        ).copy()
    )

    score_img[
        retina_mask == 0
    ] = -1

    y_grid, x_grid = np.ogrid[
        :sh,
        :sw
    ]

    distance = np.sqrt(
        (x_grid - rcx) ** 2
        +
        (y_grid - rcy) ** 2
    )

    score_img[
        distance > rr * 0.95
    ] = -1

    iy, ix = np.unravel_index(
        np.argmax(score_img),
        score_img.shape
    )

    return (
        (
            float(ix / scale),
            float(iy / scale)
        ),
        float(
            2.0 * rr / scale
        )
    )


# ============================================================
# 5. CROP HELPERS
# ============================================================

def make_square_crop(
    rgb: np.ndarray,
    center_x: float,
    center_y: float,
    crop_side: int
) -> np.ndarray:
    """
    Return a square crop.
    If crop reaches an image boundary, black padding is added.
    """

    h, w = rgb.shape[:2]

    crop_side = int(
        max(
            2,
            crop_side
        )
    )

    half = crop_side / 2.0

    x0 = int(
        round(center_x - half)
    )

    y0 = int(
        round(center_y - half)
    )

    x1 = x0 + crop_side
    y1 = y0 + crop_side

    pad_left = max(
        0,
        -x0
    )

    pad_top = max(
        0,
        -y0
    )

    pad_right = max(
        0,
        x1 - w
    )

    pad_bottom = max(
        0,
        y1 - h
    )

    if any(
        v > 0
        for v in (
            pad_left,
            pad_top,
            pad_right,
            pad_bottom
        )
    ):

        padded = cv2.copyMakeBorder(
            rgb,
            pad_top,
            pad_bottom,
            pad_left,
            pad_right,
            borderType=cv2.BORDER_CONSTANT,
            value=(0, 0, 0)
        )

        x0 += pad_left
        x1 += pad_left

        y0 += pad_top
        y1 += pad_top

        return padded[
            y0:y1,
            x0:x1
        ]

    return rgb[
        y0:y1,
        x0:x1
    ]


def prepare_model_input(
    crop_rgb: np.ndarray
) -> tuple[np.ndarray, Image.Image]:
    """
    Match the supplied glaucoma_app preprocessing:

        ImageOps.fit(..., (100,100))
        RGB
        float32 / 255
        batch dimension
    """

    pil_crop = Image.fromarray(
        crop_rgb.astype(
            np.uint8
        ),
        mode="RGB"
    )

    fitted = ImageOps.fit(
        pil_crop,
        MODEL_INPUT_SIZE,
        method=Image.Resampling.LANCZOS,
        centering=(0.5, 0.5)
    ).convert("RGB")

    arr = (
        np.asarray(
            fitted,
            dtype=np.float32
        )
        /
        255.0
    )

    batch = arr[
        np.newaxis,
        ...
    ]

    return (
        batch,
        fitted
    )


# ============================================================
# 6. INTERACTIVE REVIEW WINDOW
# ============================================================

class CropReviewer:

    def __init__(
        self,
        original_rgb: np.ndarray,
        auto_center: tuple[float, float],
        auto_side: int
    ):

        self.original = original_rgb

        self.h, self.w = (
            original_rgb.shape[:2]
        )

        # Current crop position
        self.center_x = float(
            auto_center[0]
        )

        self.center_y = float(
            auto_center[1]
        )

        # Automatic position
        self.auto_center_x = float(
            auto_center[0]
        )

        self.auto_center_y = float(
            auto_center[1]
        )

        # ----------------------------------------------------
        # Crop size limits
        # ----------------------------------------------------

        max_side = int(
            min(
                self.w,
                self.h
            )
            *
            MAX_CROP_FRACTION_OF_IMAGE
        )

        self.min_side = int(
            min(
                MIN_CROP_PIXELS,
                self.w,
                self.h
            )
        )

        self.max_side = max(
            self.min_side + 2,
            max_side
        )

        self.crop_side = int(
            np.clip(
                auto_side,
                self.min_side,
                self.max_side
            )
        )

        self.auto_side = self.crop_side

        # Mouse state
        self.dragging = False
        self.drag_start = None
        self.mouse_start_center = None

        # OpenCV window settings
        self.window_name = (
            "Optic Disc Crop Review"
        )

        self.display_w = 1180
        self.display_h = 760

    # --------------------------------------------------------
    # Display geometry
    # --------------------------------------------------------

    def _display_geometry(self):

        scale = min(
            (
                self.display_w * 0.56
            )
            /
            self.w,

            (
                self.display_h * 0.78
            )
            /
            self.h
        )

        scale = max(
            0.05,
            min(
                scale,
                1.0
            )
        )

        disp_w = max(
            1,
            int(
                round(
                    self.w * scale
                )
            )
        )

        disp_h = max(
            1,
            int(
                round(
                    self.h * scale
                )
            )
        )

        return (
            scale,
            disp_w,
            disp_h
        )

    # --------------------------------------------------------
    # Screen -> original image coordinates
    # --------------------------------------------------------

    def _screen_to_image(
        self,
        x: int,
        y: int,
        scale: float,
        left: int,
        top: int
    ):

        ix = (
            x - left
        ) / scale

        iy = (
            y - top
        ) / scale

        return (
            ix,
            iy
        )

    # --------------------------------------------------------
    # Keep crop in bounds
    # --------------------------------------------------------

    def _clamp_center(self):

        half = (
            self.crop_side
            /
            2.0
        )

        self.center_x = float(
            np.clip(
                self.center_x,
                half,
                self.w - half
            )
        )

        self.center_y = float(
            np.clip(
                self.center_y,
                half,
                self.h - half
            )
        )

    # --------------------------------------------------------
    # Generate current crop preview
    # --------------------------------------------------------

    def _crop_preview(self):

        crop = make_square_crop(
            self.original,
            self.center_x,
            self.center_y,
            self.crop_side
        )

        model_batch, fitted = (
            prepare_model_input(
                crop
            )
        )

        preview = np.asarray(
            fitted,
            dtype=np.uint8
        )

        # Enlarge 100x100 preview
        preview = cv2.resize(
            preview,
            (420, 420),
            interpolation=cv2.INTER_NEAREST
        )

        return (
            crop,
            fitted,
            preview,
            model_batch
        )

    # --------------------------------------------------------
    # Mouse callback
    # --------------------------------------------------------

    def _mouse_callback(
        self,
        event,
        x,
        y,
        flags,
        params
    ):

        scale, disp_w, disp_h = (
            self._display_geometry()
        )

        left = 25
        top = 70

        # -----------------------------------------------
        # Mouse pressed
        # -----------------------------------------------

        if event == cv2.EVENT_LBUTTONDOWN:

            ix, iy = (
                self._screen_to_image(
                    x,
                    y,
                    scale,
                    left,
                    top
                )
            )

            if (
                0 <= ix < self.w
                and
                0 <= iy < self.h
            ):

                self.dragging = True

                self.drag_start = (
                    ix,
                    iy
                )

                self.mouse_start_center = (
                    self.center_x,
                    self.center_y
                )

        # -----------------------------------------------
        # Mouse moved
        # -----------------------------------------------

        elif (
            event == cv2.EVENT_MOUSEMOVE
            and
            self.dragging
        ):

            ix, iy = (
                self._screen_to_image(
                    x,
                    y,
                    scale,
                    left,
                    top
                )
            )

            dx = (
                ix
                -
                self.drag_start[0]
            )

            dy = (
                iy
                -
                self.drag_start[1]
            )

            self.center_x = (
                self.mouse_start_center[0]
                +
                dx
            )

            self.center_y = (
                self.mouse_start_center[1]
                +
                dy
            )

            self._clamp_center()

        # -----------------------------------------------
        # Mouse released
        # -----------------------------------------------

        elif event == cv2.EVENT_LBUTTONUP:

            self.dragging = False

        # -----------------------------------------------
        # Mouse wheel
        # -----------------------------------------------

        elif event == cv2.EVENT_MOUSEWHEEL:

            if flags > 0:

                factor = 1.06

            else:

                factor = 0.94

            self.crop_side = int(
                np.clip(
                    self.crop_side * factor,
                    self.min_side,
                    self.max_side
                )
            )

    # --------------------------------------------------------
    # Run crop review
    # --------------------------------------------------------

    def run(
        self
    ) -> tuple[np.ndarray, Image.Image] | None:

        cv2.namedWindow(
            self.window_name,
            cv2.WINDOW_NORMAL
        )

        cv2.resizeWindow(
            self.window_name,
            self.display_w,
            self.display_h
        )

        cv2.setMouseCallback(
            self.window_name,
            self._mouse_callback
        )

        while True:

            scale, disp_w, disp_h = (
                self._display_geometry()
            )

            # ------------------------------------------------
            # Resize original image for display
            # ------------------------------------------------

            full_disp = cv2.resize(
                cv2.cvtColor(
                    self.original,
                    cv2.COLOR_RGB2BGR
                ),
                (
                    disp_w,
                    disp_h
                ),
                interpolation=cv2.INTER_AREA
            )

            # ------------------------------------------------
            # Crop rectangle
            # ------------------------------------------------

            half = (
                self.crop_side
                /
                2.0
            )

            x0 = int(
                round(
                    (
                        self.center_x
                        -
                        half
                    )
                    *
                    scale
                )
            )

            y0 = int(
                round(
                    (
                        self.center_y
                        -
                        half
                    )
                    *
                    scale
                )
            )

            x1 = int(
                round(
                    (
                        self.center_x
                        +
                        half
                    )
                    *
                    scale
                )
            )

            y1 = int(
                round(
                    (
                        self.center_y
                        +
                        half
                    )
                    *
                    scale
                )
            )

            cv2.rectangle(
                full_disp,
                (x0, y0),
                (x1, y1),
                (0, 165, 255),
                3
            )

            # ------------------------------------------------
            # Crop center marker
            # ------------------------------------------------

            cx_disp = int(
                round(
                    self.center_x
                    *
                    scale
                )
            )

            cy_disp = int(
                round(
                    self.center_y
                    *
                    scale
                )
            )

            cv2.drawMarker(
                full_disp,
                (
                    cx_disp,
                    cy_disp
                ),
                (0, 255, 0),
                cv2.MARKER_CROSS,
                30,
                2
            )

            # ------------------------------------------------
            # Text
            # ------------------------------------------------

            cv2.putText(
                full_disp,
                "Drag box | Mouse wheel: resize",
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            # ------------------------------------------------
            # Generate crop preview
            # ------------------------------------------------

            crop, fitted, crop_preview, _ = (
                self._crop_preview()
            )

            crop_preview_bgr = (
                cv2.cvtColor(
                    crop_preview,
                    cv2.COLOR_RGB2BGR
                )
            )

            # ------------------------------------------------
            # Canvas
            # ------------------------------------------------

            canvas_h = max(
                disp_h + 90,
                560
            )

            canvas_w = 1180

            canvas = np.zeros(
                (
                    canvas_h,
                    canvas_w,
                    3
                ),
                dtype=np.uint8
            )

            # ------------------------------------------------
            # LEFT PANE
            # ------------------------------------------------

            left_x = 25
            left_y = 70

            max_left_h = min(
                disp_h,
                canvas_h
                -
                left_y
                -
                20
            )

            if full_disp.shape[0] != max_left_h:

                ratio = (
                    max_left_h
                    /
                    full_disp.shape[0]
                )

                full_disp2 = cv2.resize(
                    full_disp,
                    (
                        max(
                            1,
                            int(
                                round(
                                    full_disp.shape[1]
                                    *
                                    ratio
                                )
                            )
                        ),
                        max_left_h
                    ),
                    interpolation=cv2.INTER_AREA
                )

            else:

                full_disp2 = full_disp

            canvas[
                left_y:
                left_y + full_disp2.shape[0],

                left_x:
                left_x + full_disp2.shape[1]
            ] = full_disp2

            # ------------------------------------------------
            # RIGHT PANE
            # ------------------------------------------------

            right_x = 700
            right_y = 80

            ph = min(
                420,
                canvas_h
                -
                right_y
                -
                20
            )

            pw = ph

            crop_preview_bgr = cv2.resize(
                crop_preview_bgr,
                (
                    pw,
                    ph
                ),
                interpolation=cv2.INTER_NEAREST
            )

            canvas[
                right_y:
                right_y + ph,

                right_x:
                right_x + pw
            ] = crop_preview_bgr

            # ------------------------------------------------
            # Right pane title
            # ------------------------------------------------

            cv2.putText(
                canvas,
                "MODEL INPUT PREVIEW",
                (right_x, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
                cv2.LINE_AA
            )

            cv2.putText(
                canvas,
                "100 x 100 RGB | normalized /255",
                (
                    right_x,
                    right_y
                    +
                    ph
                    +
                    25
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

            # ------------------------------------------------
            # Instructions
            # ------------------------------------------------

            instructions = [
                "Mouse drag = move crop",
                "Wheel or +/- = resize",
                "Arrow keys = fine movement",
                "R = reset to automatic crop",
                "ENTER = accept crop",
                "ESC = skip this image",
            ]

            y_text = (
                canvas_h
                -
                145
            )

            for i, line in enumerate(
                instructions
            ):

                cv2.putText(
                    canvas,
                    line,
                    (
                        700,
                        y_text
                        +
                        i * 22
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.50,
                    (220, 220, 220),
                    1,
                    cv2.LINE_AA
                )

            # ------------------------------------------------
            # Display
            # ------------------------------------------------

            cv2.imshow(
                self.window_name,
                canvas
            )

            key = (
                cv2.waitKey(30)
                &
                0xFF
            )

            # ------------------------------------------------
            # ENTER = accept
            # ------------------------------------------------

            if key in (
                13,
                10
            ):

                cv2.destroyWindow(
                    self.window_name
                )

                return (
                    crop,
                    fitted
                )

            # ------------------------------------------------
            # ESC = skip
            # ------------------------------------------------

            if key == 27:

                cv2.destroyWindow(
                    self.window_name
                )

                return None

            # ------------------------------------------------
            # + / =
            # ------------------------------------------------

            if key in (
                ord("+"),
                ord("=")
            ):

                self.crop_side = int(
                    np.clip(
                        self.crop_side * 1.06,
                        self.min_side,
                        self.max_side
                    )
                )

            # ------------------------------------------------
            # - / _
            # ------------------------------------------------

            elif key in (
                ord("-"),
                ord("_")
            ):

                self.crop_side = int(
                    np.clip(
                        self.crop_side * 0.94,
                        self.min_side,
                        self.max_side
                    )
                )

            # ------------------------------------------------
            # R = reset
            # ------------------------------------------------

            elif key in (
                ord("r"),
                ord("R")
            ):

                self.center_x = (
                    self.auto_center_x
                )

                self.center_y = (
                    self.auto_center_y
                )

                self.crop_side = (
                    self.auto_side
                )

                self._clamp_center()

            # ------------------------------------------------
            # LEFT ARROW
            # ------------------------------------------------

            elif key in (
                81,
                2424832
            ):

                step = max(
                    1,
                    int(
                        self.crop_side
                        *
                        0.02
                    )
                )

                self.center_x -= step

                self._clamp_center()

            # ------------------------------------------------
            # RIGHT ARROW
            # ------------------------------------------------

            elif key in (
                83,
                2555904
            ):

                step = max(
                    1,
                    int(
                        self.crop_side
                        *
                        0.02
                    )
                )

                self.center_x += step

                self._clamp_center()

            # ------------------------------------------------
            # UP ARROW
            # ------------------------------------------------

            elif key in (
                82,
                2490368
            ):

                step = max(
                    1,
                    int(
                        self.crop_side
                        *
                        0.02
                    )
                )

                self.center_y -= step

                self._clamp_center()

            # ------------------------------------------------
            # DOWN ARROW
            # ------------------------------------------------

            elif key in (
                84,
                2621440
            ):

                step = max(
                    1,
                    int(
                        self.crop_side
                        *
                        0.02
                    )
                )

                self.center_y += step

                self._clamp_center()


# ============================================================
# 7. MODEL LOADING
# ============================================================

def build_compatible_model() -> tf.keras.Model:
    """
    Fallback model definition matching the supplied training code.

    Architecture:

        input = (100,100,3)

        Conv2D:
            64
            128
            256
            512
            512

        Dense:
            64
            128
            1 sigmoid
    """

    from tensorflow.keras.models import Sequential

    from tensorflow.keras.layers import (
        Dense,
        Dropout,
        Activation,
        Flatten,
        Conv2D,
        MaxPooling2D,
        ZeroPadding2D,
        BatchNormalization,
    )

    model = Sequential()

    # --------------------------------------------------------
    # Conv block 1
    # --------------------------------------------------------

    model.add(
        Conv2D(
            64,
            (9, 9),
            input_shape=(
                100,
                100,
                3
            ),
            padding="same"
        )
    )

    model.add(
        BatchNormalization()
    )

    model.add(
        Activation("relu")
    )

    model.add(
        MaxPooling2D(
            pool_size=(2, 2)
        )
    )

    # --------------------------------------------------------
    # Conv block 2
    # --------------------------------------------------------

    model.add(
        Conv2D(
            128,
            (5, 5),
            padding="same"
        )
    )

    model.add(
        BatchNormalization()
    )

    model.add(
        Activation("relu")
    )

    model.add(
        MaxPooling2D(
            pool_size=(2, 2)
        )
    )

    # --------------------------------------------------------
    # Conv block 3
    # --------------------------------------------------------

    model.add(
        ZeroPadding2D(
            (1, 1)
        )
    )

    model.add(
        Conv2D(
            256,
            (3, 3),
            padding="same"
        )
    )

    model.add(
        BatchNormalization()
    )

    model.add(
        Activation("relu")
    )

    model.add(
        MaxPooling2D(
            pool_size=(2, 2)
        )
    )

    # --------------------------------------------------------
    # Conv block 4
    # --------------------------------------------------------

    model.add(
        ZeroPadding2D(
            (1, 1)
        )
    )

    model.add(
        Conv2D(
            512,
            (3, 3),
            padding="same"
        )
    )

    model.add(
        BatchNormalization()
    )

    model.add(
        Activation("relu")
    )

    # --------------------------------------------------------
    # Conv block 5
    # --------------------------------------------------------

    model.add(
        ZeroPadding2D(
            (1, 1)
        )
    )

    model.add(
        Conv2D(
            512,
            (3, 3),
            padding="same"
        )
    )

    model.add(
        BatchNormalization()
    )

    model.add(
        Activation("relu")
    )

    model.add(
        MaxPooling2D(
            pool_size=(2, 2)
        )
    )

    # --------------------------------------------------------
    # Fully connected
    # --------------------------------------------------------

    model.add(
        Flatten()
    )

    model.add(
        Dense(64)
    )

    model.add(
        BatchNormalization()
    )

    model.add(
        Activation("relu")
    )

    model.add(
        Dropout(0.5)
    )

    model.add(
        Dense(128)
    )

    model.add(
        BatchNormalization()
    )

    model.add(
        Activation("relu")
    )

    model.add(
        Dropout(0.5)
    )

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    model.add(
        Dense(1)
    )

    model.add(
        BatchNormalization()
    )

    model.add(
        Activation("sigmoid")
    )

    return model


def load_glaucoma_model(
    model_path: str
):

    path = Path(
        model_path
    )

    if not path.is_file():

        raise FileNotFoundError(
            f"Model file not found:\n{path}"
        )

    print(
        "\nLoading model..."
    )

    print(
        f"Model path: {path}"
    )

    # --------------------------------------------------------
    # Try direct H5 loading
    # --------------------------------------------------------

    try:

        model = (
            tf.keras.models.load_model(
                path,
                compile=False
            )
        )

        print(
            "Model loaded directly from H5."
        )

    except Exception as direct_error:

        print(
            "\nDirect H5 loading failed."
        )

        print(
            f"Reason: {direct_error}"
        )

        print(
            "Trying the exact architecture from the supplied training code..."
        )

        # ----------------------------------------------------
        # Rebuild architecture
        # ----------------------------------------------------

        model = build_compatible_model()

        model.load_weights(
            path
        )

        print(
            "Model architecture rebuilt and weights loaded successfully."
        )

    # --------------------------------------------------------
    # Check model shape
    # --------------------------------------------------------

    print(
        f"Model input shape : {model.input_shape}"
    )

    print(
        f"Model output shape: {model.output_shape}"
    )

    expected = (
        None,
        100,
        100,
        3
    )

    if tuple(
        model.input_shape
    ) != expected:

        raise ValueError(
            f"Unexpected model input shape "
            f"{model.input_shape}. "
            f"This script expects {expected}."
        )

    return model


# ============================================================
# 8. FILE DISCOVERY
# ============================================================

def discover_images(
    path: str
) -> list[Path]:

    p = Path(
        path
    )

    # --------------------------------------------------------
    # Single image
    # --------------------------------------------------------

    if p.is_file():

        if (
            p.suffix.lower()
            not in SUPPORTED_EXTENSIONS
        ):

            raise ValueError(
                f"Unsupported image extension: "
                f"{p.suffix}. "
                f"Supported: "
                f"{sorted(SUPPORTED_EXTENSIONS)}"
            )

        return [p]

    # --------------------------------------------------------
    # Folder
    # --------------------------------------------------------

    if not p.is_dir():

        raise FileNotFoundError(
            f"Image path does not exist:\n{p}"
        )

    files = [
        item
        for item in sorted(
            p.iterdir(),
            key=lambda x: x.name.lower()
        )
        if (
            item.is_file()
            and
            item.suffix.lower()
            in SUPPORTED_EXTENSIONS
        )
    ]

    if not files:

        raise FileNotFoundError(
            f"No supported images found in:\n{p}"
        )

    return files


# ============================================================
# 9. ONE-IMAGE INFERENCE
# ============================================================

def classify_one_image(
    model,
    image_path: Path
) -> str:

    print(
        "\n"
        + "=" * 78
    )

    print(
        f"IMAGE: {image_path.name}"
    )

    print(
        "=" * 78
    )

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    try:

        pil = load_rgb_image(
            image_path
        )

    except Exception as exc:

        print(
            f"ERROR: Could not read image: {exc}"
        )

        return "ERROR"

    # --------------------------------------------------------
    # RGB NumPy
    # --------------------------------------------------------

    rgb = pil_to_rgb_array(
        pil
    )

    h, w = rgb.shape[:2]

    print(
        f"Original resolution : {w} x {h}"
    )

    print(
        "Original mode       : converted/handled as RGB"
    )

    # --------------------------------------------------------
    # Automatic optic-disc detection
    # --------------------------------------------------------

    print(
        "\nDetecting optic-disc / optic-nerve-head center..."
    )

    center, retinal_diameter = (
        detect_optic_disc_center(
            rgb
        )
    )

    # --------------------------------------------------------
    # Calculate default crop
    # --------------------------------------------------------

    auto_side = int(
        round(
            retinal_diameter
            *
            AUTO_CROP_FRACTION
        )
    )

    auto_side = max(
        MIN_CROP_PIXELS,
        auto_side
    )

    auto_side_limit = int(
        min(
            w,
            h
        )
        *
        MAX_CROP_FRACTION_OF_IMAGE
    )

    auto_side = min(
        auto_side,
        max(
            MIN_CROP_PIXELS,
            auto_side_limit
        )
    )

    print(
        f"Automatic center   : "
        f"({center[0]:.1f}, {center[1]:.1f})"
    )

    print(
        f"Automatic crop     : "
        f"{auto_side} x {auto_side}"
    )

    # --------------------------------------------------------
    # Interactive crop review
    # --------------------------------------------------------

    reviewer = CropReviewer(
        original_rgb=rgb,
        auto_center=center,
        auto_side=auto_side
    )

    print(
        "\nA crop-review popup will open."
    )

    print(
        "Adjust the crop if necessary, then press ENTER."
    )

    print(
        "Press ESC in the popup to skip the image."
    )

    result = reviewer.run()

    # --------------------------------------------------------
    # User skipped
    # --------------------------------------------------------

    if result is None:

        print(
            "RESULT: SKIPPED BY USER"
        )

        return "SKIPPED"

    # --------------------------------------------------------
    # Final crop
    # --------------------------------------------------------

    crop_rgb, fitted_pil = result

    print(
        "\nFinal crop accepted from the review window."
    )

    print(
        f"Model input resolution: "
        f"{fitted_pil.size[0]} x "
        f"{fitted_pil.size[1]}"
    )

    print(
        "Model input channels  : RGB"
    )

    # --------------------------------------------------------
    # Terminal confirmation
    # --------------------------------------------------------

    while True:

        answer = input(
            "\nAllow the model to classify this image? [yes/no]: "
        ).strip().lower()

        if answer in {
            "yes",
            "y"
        }:

            break

        if answer in {
            "no",
            "n"
        }:

            print(
                "RESULT: SKIPPED BY USER"
            )

            return "SKIPPED"

        print(
            "Please type yes or no."
        )

    # --------------------------------------------------------
    # Model preprocessing
    # --------------------------------------------------------

    batch, fitted = (
        prepare_model_input(
            crop_rgb
        )
    )

    # --------------------------------------------------------
    # Run inference
    # --------------------------------------------------------

    print(
        "\nRunning model inference..."
    )

    prediction = model.predict(
        batch,
        verbose=0
    )

    # --------------------------------------------------------
    # Extract sigmoid output
    # --------------------------------------------------------

    raw_score = float(
        np.asarray(
            prediction
        ).reshape(-1)[0]
    )

    # --------------------------------------------------------
    # Based on your supplied app:
    #
    # pred > 0.5 -> Healthy
    # else       -> Glaucoma
    #
    # So:
    # raw_score > 0.5 -> No Glaucoma
    # raw_score <=0.5 -> Glaucoma
    # --------------------------------------------------------

    no_glaucoma_score = (
        raw_score
    )

    glaucoma_score = (
        1.0
        -
        raw_score
    )

    if raw_score > 0.5:

        label = (
            "No Glaucoma"
        )

    else:

        label = (
            "Glaucoma"
        )

    # --------------------------------------------------------
    # Print result
    # --------------------------------------------------------

    print(
        "\n---------- CLASSIFICATION RESULT ----------"
    )

    print(
        f"Image               : "
        f"{image_path.name}"
    )

    print(
        f"Model sigmoid score : "
        f"{raw_score:.6f}"
    )

    print(
        f"No Glaucoma score   : "
        f"{no_glaucoma_score * 100.0:.2f}%"
    )

    print(
        f"Glaucoma score      : "
        f"{glaucoma_score * 100.0:.2f}%"
    )

    print(
        f"FINAL CLASS         : "
        f"{label}"
    )

    print(
        "-------------------------------------------"
    )

    # --------------------------------------------------------
    # Optionally save final crop
    # --------------------------------------------------------

    if SAVE_ACCEPTED_CROPS:

        output_dir = (
            image_path.parent
            /
            "_reviewed_crops"
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        output_path = (
            output_dir
            /
            f"{image_path.stem}"
            f"_reviewed_100x100.png"
        )

        fitted.save(
            output_path
        )

        print(
            f"Reviewed model-input crop saved to: "
            f"{output_path}"
        )

    return label


# ============================================================
# 10. MAIN
# ============================================================

def main():

    print(
        "=" * 78
    )

    print(
        "GOLDEN-PANTHER GLAUCOMA DETECTOR - INTERACTIVE INFERENCE"
    )

    print(
        "=" * 78
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_glaucoma_model(
        MODEL_PATH
    )

    # --------------------------------------------------------
    # Discover images
    # --------------------------------------------------------

    image_files = discover_images(
        IMAGE_PATH
    )

    print(
        f"\nImages found: "
        f"{len(image_files)}"
    )

    # --------------------------------------------------------
    # Process images one by one
    # --------------------------------------------------------

    results = []

    for image_path in image_files:

        try:

            result = (
                classify_one_image(
                    model,
                    image_path
                )
            )

        except KeyboardInterrupt:

            print(
                "\nStopped by user."
            )

            break

        except Exception as exc:

            print(
                "\nERROR while processing image:"
            )

            print(
                exc
            )

            result = "ERROR"

        results.append(
            (
                image_path.name,
                result
            )
        )

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    print(
        "\n\n"
        +
        "=" * 78
    )

    print(
        "FINAL SESSION SUMMARY"
    )

    print(
        "=" * 78
    )

    if not results:

        print(
            "No images were processed."
        )

        return

    for filename, result in results:

        print(
            f"{filename:55s} -> {result}"
        )

    processed = sum(
        1
        for _, result in results
        if result in {
            "Glaucoma",
            "No Glaucoma"
        }
    )

    skipped = sum(
        1
        for _, result in results
        if result == "SKIPPED"
    )

    errors = sum(
        1
        for _, result in results
        if result == "ERROR"
    )

    print(
        "-" * 78
    )

    print(
        f"Classified : {processed}"
    )

    print(
        f"Skipped    : {skipped}"
    )

    print(
        f"Errors     : {errors}"
    )


# ============================================================
# PROGRAM START
# ============================================================

if __name__ == "__main__":
    main()