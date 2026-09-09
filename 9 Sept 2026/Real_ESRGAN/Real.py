import torch
import numpy as np

from PIL import Image, ImageFilter

from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer


# ============================================================
# MODEL
# ============================================================

model_path = 'RealESRGAN_x2plus.pth'

state_dict = torch.load(
    model_path,
    map_location=torch.device('cpu')
)['params_ema']

model = RRDBNet(
    num_in_ch=3,
    num_out_ch=3,
    num_feat=64,
    num_block=23,
    num_grow_ch=32,
    scale=2
)

model.load_state_dict(state_dict, strict=True)


# ============================================================
# REAL-ESRGAN
# ============================================================

upsampler = RealESRGANer(
    scale=2,
    model_path=model_path,
    model=model,
    tile=0,
    pre_pad=0,
    tile_pad=10,
    half=False
)


# ============================================================
# INPUT
# ============================================================

img = Image.open('edited-photo-2.png').convert('RGB')
img = np.array(img)


# ============================================================
# 2x SUPER-RESOLUTION
# ============================================================

output, _ = upsampler.enhance(
    img,
    outscale=2
)

output_img = Image.fromarray(output)


# ============================================================
# MILD SHARPENING
# ============================================================

output_img = output_img.filter(
    ImageFilter.UnsharpMask(
        radius=1.0,
        percent=130,
        threshold=3
    )
)


# ============================================================
# SAVE
# ============================================================

output_img.save('output5.png')

print("Done")