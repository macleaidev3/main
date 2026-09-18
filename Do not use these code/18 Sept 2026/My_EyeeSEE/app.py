"""
Glaucoma CDSS — HuggingFace Spaces
Input : retinal fundus image (JPEG / PNG)
Output: plain-text clinical report + downloadable PDF
"""

import os
import sys
import tempfile

import cv2
import numpy as np
import gradio as gr

sys.path.insert(0, os.path.dirname(__file__))

from phase3pipeline import Phase3Pipeline


# ── CONFIG ────────────────────────────────────────────────────────────────
REPO_ID = "Nj-1111/EyeeSEE"
EPOCH = None
TOKEN = os.getenv("HF_TOKEN_2") or os.getenv("HF_TOKEN")
# ──────────────────────────────────────────────────────────────────────────

print(f"Loading model repo={REPO_ID} epoch={'latest' if EPOCH is None else EPOCH}")

pipeline = Phase3Pipeline(
    repo_id=REPO_ID,
    epoch=EPOCH,
    mc_passes=2,
    uncertainty_threshold=0.05,
    token=TOKEN,
)

print("Model ready.")


# ── IMAGE LOADER ──────────────────────────────────────────────────────────
def _load_image(path: str) -> np.ndarray:
    img = cv2.imread(path)

    if img is None:
        raw = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(raw, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError(f"Cannot read image: {path}")

    return img


# ── REPORT BUILDER ────────────────────────────────────────────────────────
def _build_text(r: dict, mc_passes: int) -> str:
    isnt = r["isnt"]

    lines = [
        "  GLAUCOMA CDSS — CLINICAL REPORT",
        f"  vCDR              : {r['vcdr']:.4f}",
        f"  Risk Level        : {r['risk_level']}",
        f"  Uncertainty       : {r['uncertainty']:.6f}",
        f"  Sanity Check      : {'PASSED' if r['sanity_passed'] else 'CORRECTED (auto)'}",
        f"  ISNT Rule         : {'Satisfied' if isnt['rule_satisfied'] else 'Violated'}",
        "",
        "  ISNT Rim Thickness",
        f"    Inferior  : {isnt['inferior']:.2f}",
        f"    Superior  : {isnt['superior']:.2f}",
        f"    Nasal     : {isnt['nasal']:.2f}",
        f"    Temporal  : {isnt['temporal']:.2f}",
        "",
        "  Structural",
        f"    Disc Area : {r['disc_area_px']:,} px",
        f"    Cup Area  : {r['cup_area_px']:,} px",
        f"    Cup/Disc  : {r['cup_area_px'] / max(r['disc_area_px'], 1) * 100:.1f}%",
        f"    MC Passes : {mc_passes}",
    ]
    

    warnings = r.get("warnings", [])
    if warnings:
        lines += ["", "  Warnings"]
        for w in warnings:
            lines.append(f"    ! {w}")

    

    return "\n".join(lines)


# ── INFERENCE ─────────────────────────────────────────────────────────────
def analyse(file_path):
    if file_path is None:
        return "No file uploaded.", None, "Please upload a JPEG or PNG fundus image."

    try:
        print(f"Received file: {file_path}")

        img = _load_image(file_path)
        print(f"Image loaded successfully. Shape: {img.shape}")

        result = pipeline.run(img)
        print("Pipeline inference completed.")

        text = _build_text(result["report"], pipeline.mc_passes)
        status = "Analysis completed successfully."
        

        return text, status

    except Exception as e:
        print(f"Analysis error: {e}")
        return f"Error: {e}", None, f"Analysis failed: {e}"


def set_busy():
    return gr.update(interactive=False), "Running analysis, please wait..."


def set_ready():
    return gr.update(interactive=True)


# ── UI ────────────────────────────────────────────────────────────────────
with gr.Blocks(title="Glaucoma CDSS") as demo:
    gr.Markdown(
        "## Glaucoma CDSS\n"
        "Upload a retinal fundus image to receive a clinical screening report."
    )

    with gr.Row():
        file_in = gr.File(
            label="Fundus Image (JPEG / PNG)",
            file_types=[".jpg", ".jpeg", ".png"],
            type="filepath",
        )

        run_btn = gr.Button("Analyse", variant="primary")

    status_box = gr.Textbox(
        label="Status",
        value="Awaiting image upload.",
        interactive=False,
    )

    report_box = gr.Textbox(
        label="Clinical Report",
        lines=28,
        interactive=False,
    )


    gr.Markdown(
        "_Research prototype — NOT a medical device. "
        "All results must be reviewed by a qualified ophthalmologist._"
    )
    gr.Markdown(
    """
# Glaucoma CDSS
--

## How to Interpret the Report

### 1. vCDR (Vertical Cup-to-Disc Ratio)
The most important glaucoma screening metric.
General interpretation:
- **0.30 – 0.50** → Usually within healthy range
- **0.50 – 0.65** → Borderline / monitor carefully
- **0.65 – 0.80** → Glaucoma suspect
- **> 0.80** → High glaucoma risk

Higher values indicate enlargement of the optic cup relative to the optic disc.

---

## 3. Uncertainty Score
Represents model confidence.
- **< 0.05** → Stable prediction
- **0.05 – 0.10** → Moderate uncertainty
- **> 0.10** → Low confidence prediction

High uncertainty may occur with:
- poor image quality,
- blur,
- extreme lighting,
- incomplete optic disc visibility.

---

## 4. Structural Measurements

### Disc Area
Estimated optic disc size in pixels.
### Cup Area
Estimated optic cup size in pixels.
### Cup/Disc %
Percentage of cup area relative to disc area.
Larger cup proportions may indicate glaucomatous damage.

---

## 5. Risk Levels

### Healthy
No major structural glaucoma indicators detected.
### Glaucoma Suspect
One or more warning signs detected:
- elevated vCDR,
- ISNT violation,
- anatomical inconsistency,
- or uncertain segmentation.

### High Risk
Strong structural indicators of glaucoma detected.
Clinical ophthalmology review is strongly recommended.

---

## Important Disclaimer
This system is a **research prototype** and NOT a certified medical device.
The generated report is intended for:
- educational use,
- AI research,
- and preliminary screening assistance only.

All clinical decisions must be made by a qualified ophthalmologist.
"""
)

    run_btn.click(
        fn=set_busy,
        inputs=None,
        outputs=[run_btn, status_box],
        queue=False,
    ).then(
        fn=analyse,
        inputs=[file_in],
        outputs=[report_box, status_box],
    ).then(
        fn=set_ready,
        inputs=None,
        outputs=[run_btn],
        queue=False,
    )

if __name__ == "__main__":
    demo.launch()