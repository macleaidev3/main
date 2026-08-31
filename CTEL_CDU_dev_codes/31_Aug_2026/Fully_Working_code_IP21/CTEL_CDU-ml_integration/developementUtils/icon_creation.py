from PIL import Image

# PNG logo
input_png = "logo.png"

# Output ICO
output_ico = "logo.ico"

# Standard Windows icon sizes
icon_sizes = [
    (16, 16),
    (20, 20),
    (24, 24),
    (32, 32),
    (40, 40),
    (48, 48),
    (64, 64),
    (96, 96),
    (128, 128),
    (256, 256),
]

img = Image.open(input_png).convert("RGBA")

# Save all resolutions into a single ICO file
img.save(
    output_ico,
    format="ICO",
    sizes=icon_sizes,
)

print(f"Icon saved as {output_ico}")