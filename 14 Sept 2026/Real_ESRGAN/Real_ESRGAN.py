import torch
import numpy as np
from PIL import Image, ImageFilter
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from pathlib import Path


# ====================== FOLDERS =======================================

# Folder where this Python script is located
BASE_DIR = Path(__file__).resolve().parent

# Original images folder
INPUT_FOLDER = BASE_DIR / "Original_images"

# Result folder
OUTPUT_FOLDER = BASE_DIR / "ESRGAN_Result"

# Create Result folder if it doesn't exist
OUTPUT_FOLDER.mkdir(exist_ok=True)


# ================== MODEL ==============================================

model_path = BASE_DIR / "RealESRGAN_x2plus.pth"

state_dict = torch.load(model_path, map_location=torch.device("cpu"))["params_ema"]
model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=2)
model.load_state_dict(state_dict, strict=True)

# =====================REAL-ESRGAN =======================================
upsampler = RealESRGANer(scale=2, model_path=str(model_path), model=model, tile=0, pre_pad=0, tile_pad=10, half=False)

# ===================FIND ALL IMAGES ===================================
supported_extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp"
}

image_files = [
    file for file in INPUT_FOLDER.iterdir()
    if file.is_file() and file.suffix.lower() in supported_extensions
]

# =======================CHECK IF IMAGES EXIST============================================================
if not image_files:
    print("No images found in:")
    print(INPUT_FOLDER)
    exit()
print(f"Found {len(image_files)} image(s).")
print()

# =====================PROCESS ALL IMAGES ============================================================
for index, image_path in enumerate(image_files, start=1):

    print(f"[{index}/{len(image_files)}] Processing: {image_path.name}")

    try:

        # ------------------- LOAD IMAGE ----------------------------------------------------
        img = Image.open(image_path).convert("RGB")
        img = np.array(img)

        # -------------------- 2X SUPER-RESOLUTION ----------------------------------------------------
        output, _ = upsampler.enhance(img,  outscale=2)
        output_img = Image.fromarray(output)

        # --------------------- MILD SHARPENING ----------------------------------------------------
        output_img = output_img.filter(ImageFilter.UnsharpMask(radius=1.0, percent=130, threshold=3))


        # ---------------------- SAVE WITH SAME FILENAME ----------------------------------------------------
        output_path = OUTPUT_FOLDER / image_path.name
        output_img.save(output_path)
        print(f"Saved: {output_path}")

    except Exception as e:
        print(f"ERROR processing {image_path.name}")
        print(f"{e}")

# ====================== COMPLETE ============================================================
print()
print("==============================================")
print("All images have been processed.")
print(f"Results are saved in: {OUTPUT_FOLDER}")
print("==============================================")