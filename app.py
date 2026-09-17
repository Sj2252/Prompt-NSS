"""
app.py - AI Prompt Competition Scoring Tool
Evaluates player-generated images against target reference images using
OpenAI's CLIP model (clip-vit-base-patch32) and token efficiency penalty.
Includes automatic package checking and installation on startup.
"""

import os
import sys
import subprocess
import importlib.util

# Ensure utf-8 console output on Windows
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# ---------------------------------------------------------------------------
# 0. Automatic Dependency Check & Auto-Installation
# ---------------------------------------------------------------------------
REQUIRED_PACKAGES = {
    "torch": "torch",
    "torchvision": "torchvision",
    "transformers": "transformers",
    "gradio": "gradio",
    "PIL": "pillow",
    "numpy": "numpy",
    "accelerate": "accelerate"
}

def ensure_dependencies():
    """Checks for required packages and automatically installs missing ones."""
    missing = []
    for import_name, pkg_name in REQUIRED_PACKAGES.items():
        if importlib.util.find_spec(import_name) is None:
            missing.append(pkg_name)

    if missing:
        print(f"[!] Detected missing dependencies: {', '.join(missing)}")
        print("[*] Automatically installing missing packages... Please wait.")
        try:
            cmd = [sys.executable, "-m", "pip", "install"] + missing
            subprocess.check_call(cmd)
            print("[OK] All missing dependencies installed successfully!\n")
        except subprocess.CalledProcessError as err:
            print(f"[ERROR] Auto-installation error: {err}")
            print("[!] Please run: pip install -r requirements.txt manually.")
            sys.exit(1)

# Run check before heavy imports
ensure_dependencies()

# ---------------------------------------------------------------------------
# Core Imports (Post-dependency check)
# ---------------------------------------------------------------------------
import torch
import numpy as np
from PIL import Image
import gradio as gr
from transformers import CLIPProcessor, CLIPModel

# ---------------------------------------------------------------------------
# Device & Model Initialization
# ---------------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_ID = "openai/clip-vit-base-patch32"

print(f"[*] Initializing AI Prompt Scoring Engine on device: {DEVICE}")
model = CLIPModel.from_pretrained(MODEL_ID).to(DEVICE)
processor = CLIPProcessor.from_pretrained(MODEL_ID)
model.eval()
print(f"[OK] Successfully loaded {MODEL_ID} on {DEVICE}")


# ---------------------------------------------------------------------------
# Core Scoring Logic
# ---------------------------------------------------------------------------
def compute_clip_similarity(ref_img: Image.Image, gen_img: Image.Image) -> float:
    """
    Computes visual cosine similarity between two PIL images using CLIP.
    Returns float in range [0.0, 1.0].
    """
    ref_rgb = ref_img.convert("RGB")
    gen_rgb = gen_img.convert("RGB")

    inputs = processor(images=[ref_rgb, gen_rgb], return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        features = model.get_image_features(**inputs)
        # In transformers 5.x, get_image_features returns BaseModelOutputWithPooling
        if hasattr(features, "pooler_output") and features.pooler_output is not None:
            features = features.pooler_output
        elif hasattr(features, "image_embeds") and features.image_embeds is not None:
            features = features.image_embeds
        elif isinstance(features, (tuple, list)):
            features = features[1] if len(features) > 1 else features[0]

        # Normalize embeddings to unit length (L2 norm)
        normalized = features / features.norm(p=2, dim=-1, keepdim=True)
        # Cosine similarity is dot product of normalized vectors
        cos_sim = torch.cosine_similarity(normalized[0:1], normalized[1:2]).item()

    # Clip similarity to non-negative range [0.0, 1.0]
    return max(0.0, min(1.0, float(cos_sim)))


def calculate_score(
    ref_img: Image.Image,
    gen_img: Image.Image,
    token_count: float,
    lambda_weight: float
):
    """
    Evaluates submission and returns visual similarity %, token deduction, final score,
    and a formatted markdown analysis report.
    """
    if ref_img is None:
        raise gr.Error("Please upload or select a Target Reference Image.")
    if gen_img is None:
        raise gr.Error("Please upload or select a Player's Generated Image.")

    tokens = max(0, int(round(token_count or 0)))
    lam = max(0.0, float(lambda_weight if lambda_weight is not None else 0.3))

    # 1. Visual Similarity
    raw_sim = compute_clip_similarity(ref_img, gen_img)
    similarity_pct = round(raw_sim * 100.0, 2)

    # 2. Token Penalty Deduction
    deduction = round(lam * tokens, 2)

    # 3. Final Score: Final Score = (Similarity %) - (Lambda * Tokens)
    final_score = round(similarity_pct - deduction, 2)

    return similarity_pct, deduction, final_score


# ---------------------------------------------------------------------------
# Ellipsus-Inspired Editorial Theme & Stylesheet (Design Specification)
# ---------------------------------------------------------------------------
custom_css = """
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;0,6..72,600;1,6..72,400&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap');

:root, body, .dark, .gradio-container {
    --bg-canvas: #141311;
    --bg-surface: #1C1B18;
    --bg-inset: #252420;
    --bg-hover: #2E2C27;
    --border-strong: #33312B;
    --border-subtle: #23221E;
    --text-primary: #F4EFE6;
    --text-muted: #ABA598;
    --text-faint: #736F66;
    --accent-sage: #5E826B;
    --accent-rust: #C86843;
    --action-bg: #F4EFE6;
    --action-fg: #141311;

    /* Gradio theme variable mappings */
    --body-background-fill: var(--bg-canvas) !important;
    --background-fill-primary: var(--bg-surface) !important;
    --background-fill-secondary: var(--bg-inset) !important;
    --block-background-fill: var(--bg-inset) !important;
    --block-border-color: var(--border-strong) !important;
    --block-border-width: 1px !important;
    --border-color-primary: var(--border-strong) !important;
    --border-color-accent: var(--text-muted) !important;
    --body-text-color: var(--text-primary) !important;
    --block-label-text-color: var(--text-muted) !important;
    --block-title-text-color: var(--text-primary) !important;
    --input-background-fill: var(--bg-inset) !important;
    --input-border-color: var(--border-strong) !important;
    --input-border-color-focus: var(--text-muted) !important;
    --button-primary-background-fill: var(--action-bg) !important;
    --button-primary-background-fill-hover: #E2DDD4 !important;
    --button-primary-text-color: var(--action-fg) !important;
    --button-secondary-background-fill: transparent !important;
    --button-secondary-background-fill-hover: var(--bg-inset) !important;
    --button-secondary-text-color: var(--text-muted) !important;
    --button-secondary-border-color: var(--border-strong) !important;
    --slider-color: var(--text-muted) !important;
}

/* Base typography & page resets */
html, body {
    background-color: var(--bg-canvas) !important;
    color: var(--text-primary) !important;
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
    margin: 0;
    padding: 0;
    -webkit-font-smoothing: antialiased;
}

.gradio-container {
    max-width: 1240px !important;
    margin: 0 auto !important;
    padding: 28px 20px !important;
    background-color: var(--bg-canvas) !important;
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif !important;
}

/* Editorial Header (Warm carbon container with editorial serif title) */
.editorial-header {
    background-color: var(--bg-surface);
    border: 1px solid var(--border-strong);
    border-radius: 8px;
    padding: 20px 24px;
    margin-bottom: 24px;
    box-shadow: none;
}

.header-inner {
    display: flex;
    flex-direction: column;
    gap: 8px;
}

.header-title-row {
    display: flex;
    align-items: center;
    gap: 12px;
}

.header-title {
    font-family: 'Newsreader', 'Spectral', 'Georgia', serif !important;
    font-size: 1.4rem;
    font-weight: 500;
    letter-spacing: -0.02em;
    color: var(--text-primary) !important;
    margin: 0;
    line-height: 1.25;
}

/* Editorial Monospace Tag */
.editorial-tag {
    display: inline-flex;
    align-items: center;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    color: var(--text-muted);
    background-color: var(--bg-inset);
    border: 1px solid var(--border-strong);
    padding: 2px 8px;
    border-radius: 4px;
    letter-spacing: 0.02em;
    white-space: nowrap;
}

/* Two-column layout on desktop, stacking under 768px */
.main-grid {
    display: flex !important;
    flex-direction: row !important;
    gap: 24px !important;
    align-items: stretch !important;
}

.card-panel {
    flex: 1 1 0% !important;
    background-color: var(--bg-surface) !important;
    border: 1px solid var(--border-strong) !important;
    border-radius: 8px !important;
    padding: 24px !important;
    box-shadow: none !important;
    box-sizing: border-box !important;
}

@media (max-width: 768px) {
    .main-grid {
        flex-direction: column !important;
        gap: 20px !important;
    }
    .card-panel {
        width: 100% !important;
    }
}

/* Section Header inside Cards */
.section-header {
    display: flex;
    align-items: center;
    gap: 8px;
    margin-bottom: 20px;
    padding-bottom: 12px;
    border-bottom: 1px solid var(--border-subtle);
}

.section-icon {
    color: var(--text-muted);
    flex-shrink: 0;
}

.section-title {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-size: 0.9rem;
    font-weight: 600;
    letter-spacing: -0.01em;
    color: var(--text-primary);
    flex-grow: 1;
}

/* Image comparison frames: chromatic-neutral backplate with subtle warm checker */
.image-container,
[data-testid="image"],
.drop-zone,
.upload-container,
.image-uploader {
    background-color: var(--bg-canvas) !important;
    background-image: 
        linear-gradient(45deg, #181714 25%, transparent 25%), 
        linear-gradient(-45deg, #181714 25%, transparent 25%), 
        linear-gradient(45deg, transparent 75%, #181714 75%), 
        linear-gradient(-45deg, transparent 75%, #181714 75%) !important;
    background-size: 16px 16px !important;
    background-position: 0 0, 0 8px, 8px -8px, -8px 0px !important;
    border: 1px solid var(--border-strong) !important;
    border-radius: 6px !important;
    overflow: hidden !important;
    box-shadow: none !important;
    transition: border-color 0.15s ease !important;
}

.image-container:hover,
[data-testid="image"]:hover,
.drop-zone:hover,
.upload-container:hover {
    border-color: var(--text-muted) !important;
}

/* Inputs / Number boxes */
input[type="number"],
input[type="text"],
textarea {
    background-color: var(--bg-inset) !important;
    border: 1px solid var(--border-strong) !important;
    border-radius: 6px !important;
    color: var(--text-primary) !important;
    font-family: 'JetBrains Mono', monospace !important;
    padding: 10px 12px !important;
    box-shadow: none !important;
    outline: none !important;
    font-size: 0.9rem !important;
    transition: border-color 0.15s ease !important;
}

input[type="number"]:focus,
input[type="text"]:focus,
textarea:focus {
    border-color: var(--text-muted) !important;
    outline: none !important;
    box-shadow: none !important;
}

/* Labels typography */
label span,
.block-title,
.gr-form label {
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-size: 13px !important;
    font-weight: 500 !important;
    color: var(--text-muted) !important;
}

/* Primary Button: High-contrast inverted background */
button.primary,
.btn-primary {
    background-color: var(--action-bg) !important;
    color: var(--action-fg) !important;
    border: 1px solid var(--action-bg) !important;
    border-radius: 6px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.875rem !important;
    padding: 8px 16px !important;
    box-shadow: none !important;
    transform: none !important;
    transition: background-color 0.15s ease, border-color 0.15s ease, color 0.15s ease !important;
    cursor: pointer !important;
}

button.primary:hover,
.btn-primary:hover {
    background-color: #E2DDD4 !important;
    border-color: #E2DDD4 !important;
    color: #141311 !important;
    box-shadow: none !important;
    transform: none !important;
}

/* Secondary Button: Subtle perimeter border */
button.secondary,
.btn-secondary,
button:not(.primary) {
    background-color: transparent !important;
    color: var(--text-muted) !important;
    border: 1px solid var(--border-strong) !important;
    border-radius: 6px !important;
    font-family: 'Plus Jakarta Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.875rem !important;
    padding: 8px 16px !important;
    box-shadow: none !important;
    transform: none !important;
    transition: background-color 0.15s ease, border-color 0.15s ease, color 0.15s ease !important;
    cursor: pointer !important;
}

button.secondary:hover,
.btn-secondary:hover,
button:not(.primary):hover {
    background-color: var(--bg-inset) !important;
    color: var(--text-primary) !important;
    border-color: var(--border-strong) !important;
    box-shadow: none !important;
    transform: none !important;
}

/* Slider Controls */
input[type="range"] {
    accent-color: var(--text-muted) !important;
}

input[type="range"]::-webkit-slider-runnable-track {
    background: var(--bg-inset) !important;
    border: 1px solid var(--border-strong) !important;
    border-radius: 4px !important;
    height: 6px !important;
}

input[type="range"]::-webkit-slider-thumb {
    background: var(--text-primary) !important;
    border-radius: 50% !important;
    border: none !important;
    box-shadow: none !important;
    cursor: pointer !important;
    width: 14px !important;
    height: 14px !important;
    margin-top: -5px !important;
}

input[type="range"]::-moz-range-track {
    background: var(--bg-inset) !important;
    border: 1px solid var(--border-strong) !important;
    border-radius: 4px !important;
    height: 6px !important;
}

input[type="range"]::-moz-range-thumb {
    background: var(--text-primary) !important;
    border-radius: 50% !important;
    border: none !important;
    box-shadow: none !important;
    cursor: pointer !important;
    width: 14px !important;
    height: 14px !important;
}

input[type="range"]::-moz-range-progress {
    background: var(--text-muted) !important;
    height: 6px !important;
    border-radius: 4px !important;
}

/* Metric Display Formatting */
.sim-box input {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.15rem !important;
    font-weight: 500 !important;
    color: var(--text-primary) !important;
}

.deduct-box input {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.15rem !important;
    font-weight: 600 !important;
    color: var(--accent-rust) !important;
}

.score-box input {
    font-family: 'Newsreader', 'Spectral', 'Georgia', serif !important;
    font-size: 2.2rem !important;
    font-weight: 500 !important;
    letter-spacing: -0.02em !important;
    color: var(--text-primary) !important;
}

/* Scoring Logic Box */
.logic-box {
    background-color: rgba(37, 36, 32, 0.6);
    border: 1px solid var(--border-strong);
    color: var(--text-muted);
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.75rem;
    border-radius: 6px;
    padding: 14px 16px;
    margin-top: 18px;
    line-height: 1.5;
}

.logic-title {
    font-family: 'Plus Jakarta Sans', sans-serif;
    font-weight: 600;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    color: var(--text-muted);
    margin-bottom: 8px;
}

.logic-expression {
    color: var(--text-primary);
    background: #1C1B18;
    border: 1px solid var(--border-strong);
    padding: 8px 12px;
    border-radius: 4px;
    margin-bottom: 0;
    font-size: 0.82rem;
    font-weight: 500;
}

/* Completely remove Gradio watermark, footer links, and settings */
footer, .footer {
    display: none !important;
    visibility: hidden !important;
    height: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
}
"""

editorial_theme = gr.themes.Base(
    primary_hue=gr.themes.colors.neutral,
    neutral_hue=gr.themes.colors.stone,
    font=["Plus Jakarta Sans", "-apple-system", "BlinkMacSystemFont", "Segoe UI", "Roboto", "sans-serif"],
    font_mono=["JetBrains Mono", "ui-monospace", "SFMono-Regular", "Menlo", "Monaco", "Consolas", "monospace"],
).set(
    body_background_fill="#141311",
    body_background_fill_dark="#141311",
    block_background_fill="#252420",
    block_background_fill_dark="#252420",
    block_border_color="#33312B",
    block_border_color_dark="#33312B",
    border_color_primary="#33312B",
    border_color_primary_dark="#33312B",
    body_text_color="#F4EFE6",
    body_text_color_dark="#F4EFE6",
    block_label_text_color="#ABA598",
    block_label_text_color_dark="#ABA598",
    input_background_fill="#252420",
    input_background_fill_dark="#252420",
    button_primary_background_fill="#F4EFE6",
    button_primary_background_fill_hover="#E2DDD4",
    button_primary_text_color="#141311",
    button_secondary_background_fill="transparent",
    button_secondary_text_color="#ABA598",
    button_secondary_border_color="#33312B",
    slider_color="#ABA598",
)

with gr.Blocks(title="Image Comparison & Scoring Tool") as demo:
    gr.HTML("""
    <header class="editorial-header">
        <div class="header-inner">
            <div class="header-title-row">
                <h1 class="header-title">Image Comparison &amp; Scoring</h1>
                <span class="editorial-tag">CLIP ViT-B/32</span>
            </div>
        </div>
    </header>
    """)

    with gr.Row(elem_classes=["main-grid"]):
        # Left Column: Inputs
        with gr.Column(scale=1, elem_classes=["card-panel"]):
            gr.HTML("""
            <div class="section-header">
                <svg class="section-icon" xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M21 12H3"></path>
                    <path d="M9 6v12"></path>
                    <path d="M15 6v12"></path>
                </svg>
                <span class="section-title">Submission Inputs</span>
            </div>
            """)

            with gr.Row():
                ref_input = gr.Image(
                    label="Target Reference Image",
                    type="pil",
                    height=260
                )
                gen_input = gr.Image(
                    label="Player Generated Image",
                    type="pil",
                    height=260
                )

            tokens_input = gr.Number(
                label="Prompt Token Count",
                value=15,
                precision=0,
                minimum=0,
                step=1
            )
            lambda_input = gr.Slider(
                label="Penalty Weight (λ)",
                minimum=0.0,
                maximum=1.0,
                value=0.3,
                step=0.05
            )

            with gr.Row():
                calc_btn = gr.Button("Calculate Score", variant="primary", size="lg")
                clear_btn = gr.ClearButton(
                    components=[ref_input, gen_input, tokens_input, lambda_input],
                    value="Clear Inputs",
                    variant="secondary"
                )

        # Right Column: Outputs
        with gr.Column(scale=1, elem_classes=["card-panel"]):
            gr.HTML("""
            <div class="section-header">
                <svg class="section-icon" xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <line x1="18" x2="18" y1="20" y2="10"></line>
                    <line x1="12" x2="12" y1="20" y2="4"></line>
                    <line x1="6" x2="6" y1="20" y2="14"></line>
                </svg>
                <span class="section-title">Evaluation Results</span>
                <span class="editorial-tag">Metric Breakdown</span>
            </div>
            """)

            sim_output = gr.Number(
                label="Visual Similarity (%)",
                precision=2,
                interactive=False,
                elem_classes=["sim-box"]
            )
            deduct_output = gr.Number(
                label="Token Deduction (-)",
                precision=2,
                interactive=False,
                elem_classes=["deduct-box"]
            )
            score_output = gr.Number(
                label="Final Score",
                precision=2,
                interactive=False,
                elem_classes=["score-box"]
            )

            gr.HTML("""
            <div class="logic-box">
                <div class="logic-title">Scoring Definition</div>
                <div class="logic-expression">Final Score = Visual Similarity (%) − (λ × Tokens)</div>
            </div>
            """)

    # Wire event handler
    calc_btn.click(
        fn=calculate_score,
        inputs=[ref_input, gen_input, tokens_input, lambda_input],
        outputs=[sim_output, deduct_output, score_output]
    )

if __name__ == "__main__":
    head_script = """
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600&family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;0,6..72,600;1,6..72,400&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
    <script>document.documentElement.classList.add("dark"); document.body.classList.add("dark");</script>
    """

    # Attempt port 7860, automatically falling back to subsequent ports if occupied
    launched = False
    for port in range(7860, 7875):
        try:
            demo.launch(
                server_name="127.0.0.1",
                server_port=port,
                share=False,
                theme=editorial_theme,
                css=custom_css,
                head=head_script,
                footer_links=[]
            )
            launched = True
            break
        except OSError as err:
            if "port" in str(err).lower():
                print(f"[!] Port {port} is occupied, trying port {port + 1}...")
                continue
            raise err

    if not launched:
        demo.launch(
            server_name="127.0.0.1",
            share=False,
            theme=editorial_theme,
            css=custom_css,
            head=head_script,
            footer_links=[]
        )

