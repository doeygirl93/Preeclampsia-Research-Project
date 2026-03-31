import gradio as gr
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import cv2
import os
import time

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_IMAGE_SIZE = (224, 224)
MODEL_PATH = 'hypertensive_fine_tuned_best_model.pth'
DECISION_THRESHOLD = 0.0421
UNCERTAINTY_LOW = 0.45
UNCERTAINTY_HIGH = 0.55
MIN_RESOLUTION = 168
BLUR_THRESHOLD = 20

TRANSFORMS_TENSOR = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

target_activation = None
target_grad = None


def save_activation(module, input, output):
    global target_activation
    target_activation = output.detach()


def save_gradient(module, grad_input, grad_output):
    global target_grad
    target_grad = grad_output[0].detach()


def load_model():
    model = models.densenet201(weights=None)
    model.classifier = nn.Linear(model.classifier.in_features, 1)
    if os.path.exists(MODEL_PATH):
        try:
            state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
            model.load_state_dict(state_dict)
            status = "Model sucessfully Loaded....."
        except Exception as e:
            status = f"Error loading weights: {e}"
    else:
        status = f" Weights not found at {MODEL_PATH}"
    model.to(DEVICE)
    model.eval()
    return model, status


model, GLOBAL_STATUS = load_model()


def check_image_quality(img_rgb):
    h, w = img_rgb.shape[:2]
    if h < MIN_RESOLUTION or w < MIN_RESOLUTION:
        return False, f"IMAGE IS REJECTED: The resolution is too low for the screening. ({w}×{h} -- minimum of {MIN_RESOLUTION}×{MIN_RESOLUTION} is required)"

    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
    if blur_score < BLUR_THRESHOLD:
        return False, f"IMAGE IS REJECTED: Image is too blurry. Please stabilise and retake for accarate screening. (Laplacian variance: {blur_score:.1f} -- minimum of {BLUR_THRESHOLD} required)"

    return True, None


def preprocess(img_rgb):
    img_resized = cv2.resize(img_rgb, INPUT_IMAGE_SIZE)
    tensor = TRANSFORMS_TENSOR(img_resized).unsqueeze(0).to(DEVICE)
    return tensor


def generate_gradcam(input_tensor, original_img_rgb):
    global target_activation, target_grad
    target_activation, target_grad = None, None

    # Fix: Accessing the last layer of the block safely
    layers = list(model.features.denseblock4.children())
    target_layer = layers[-2].conv2

    h1 = target_layer.register_forward_hook(save_activation)
    h2 = target_layer.register_full_backward_hook(save_gradient)

    logits = model(input_tensor)
    prob = torch.sigmoid(logits).item()
    model.zero_grad()
    logits.backward()

    # Clean up hooks immediately
    h1.remove()
    h2.remove()

    if target_activation is not None and target_grad is not None:
        # Focusing on positive gradients for better specificity
        weights = torch.mean(torch.clamp(target_grad, min=0), dim=(2, 3), keepdim=True)
        cam = torch.sum(weights * target_activation, dim=1).squeeze().detach().cpu().numpy()

        cam = np.maximum(cam, 0)

        # Safety check to avoid division by zero
        denom = (cam.max() - cam.min())
        if denom != 0:
            cam = (cam - cam.min()) / denom

        # Specificity Thresholding: Ignore weak background "blobs"
        cam[cam < 0.2] = 0

        h, w = original_img_rgb.shape[:2]
        heatmap = cv2.resize(cam, (w, h))
        heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        # Blend heatmap with original image
        blended = cv2.addWeighted(original_img_rgb, 0.6, heatmap_color, 0.4, 0)
        return blended, prob

    return original_img_rgb, prob


def predict_and_explain(input_img):
    if input_img is None:
        empty_html = "<div style='padding:20px;color:#64748b;text-align:center;'>Upload A Retinal Fundus Image To Begin Analysis.</div>"
        return None, empty_html, ""

    t_start = time.perf_counter()

    passed, rejection_msg = check_image_quality(input_img)
    if not passed:
        rejection_html = f"""
        <div style='padding:16px;border-radius:10px;background:#fef2f2;border:1px solid #fca5a5;text-align:center;'>
            <p style='font-size:1.1em;color:#b91c1c;font-weight:600;margin:0;'>{rejection_msg}</p>
        </div>"""
        return input_img, rejection_html, ""

    input_tensor = preprocess(input_img)
    blended, prob = generate_gradcam(input_tensor, input_img)

    t_end = time.perf_counter()
    inference_ms = (t_end - t_start) * 1000

    if UNCERTAINTY_LOW <= prob <= UNCERTAINTY_HIGH:
        badge_color = "#f59e0b"
        badge_bg = "#fffbeb"
        badge_border = "#fcd34d"
        status_line = "UNCERTAIN"
        sub_line = "The data is a boardline case and just near the threshold. A manual review is heavily advised."
    elif prob >= DECISION_THRESHOLD:
        badge_color = "#dc2626"
        badge_bg = "#fef2f2"
        badge_border = "#fca5a5"
        status_line = "HIGH RISK"
        sub_line = "Hypertensive retinopathy was sucessfuly detected."
    else:
        badge_color = "#16a34a"
        badge_bg = "#f0fdf4"
        badge_border = "#86efac"
        status_line = "LOW RISK"
        sub_line = "No significant hypertensive retinopathy signed were found."

    result_html = f"""
    <div style='font-family:system-ui,sans-serif;'>
        <div style='padding:18px 20px;border-radius:10px;background:{badge_bg};border:1.5px solid {badge_border};text-align:center;margin-bottom:12px;'>
            <p style='font-size:1.5em;font-weight:700;color:{badge_color};margin:0 0 4px 0;'>{status_line}</p>
            <p style='font-size:0.95em;color:#475569;margin:0;'>{sub_line}</p>
        </div>
        <div style='display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-bottom:12px;'>
            <div style='background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:12px;text-align:center;'>
                <p style='font-size:0.75em;color:#94a3b8;margin:0 0 2px 0;text-transform:uppercase;letter-spacing:.05em;'>Probability Score</p>
                <p style='font-size:1.4em;font-weight:700;color:#1e293b;margin:0;'>{prob:.4f}</p>
            </div>
            <div style='background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:12px;text-align:center;'>
                <p style='font-size:0.75em;color:#94a3b8;margin:0 0 2px 0;text-transform:uppercase;letter-spacing:.05em;'>The Decision Threshold</p>
                <p style='font-size:1.4em;font-weight:700;color:#1e293b;margin:0;'>{DECISION_THRESHOLD}</p>
            </div>
        </div>
        <div style='background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;padding:10px 14px;'>
            <p style='font-size:0.8em;color:#64748b;margin:0;'>
                ⏱ Prediction Time: <b>{inference_ms:.1f} ms</b> &nbsp;|&nbsp;
                Device: <b>{str(DEVICE).upper()}</b> &nbsp;|&nbsp;
                Image: <b>{input_img.shape[1]}×{input_img.shape[0]}px</b>
            </p>
        </div>
    </div>
    """

    return blended, result_html, ""


CSS = """
.gradio-container { max-width: 1100px !important; margin: auto; }
footer { display: none !important; }
#disclaimer {
    text-align: center;
    color: #94a3b8;
    font-size: 0.78em;
    margin-top: 8px;
    padding: 8px;
    border-top: 1px solid #e2e8f0;
}
"""

with gr.Blocks(theme=gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"]
), css=CSS, title="Preeclampsia Screening Tool") as demo:
    gr.Markdown(
        """
        # Preeclampsia Screening via Hypertensive Retinopathy
        ### AI driven retinal fundus based image analysis to detect PE -By Chika
        """
    )
    gr.Markdown(f"**System status:** {GLOBAL_STATUS} &nbsp;|&nbsp; **Device:** {str(DEVICE).upper()}")

    gr.Markdown("---")

    with gr.Row(equal_height=True):
        with gr.Column(scale=1):
            gr.Markdown("### Input")
            input_view = gr.Image(
                label="Upload Fundus Image here",
                type="numpy",
                image_mode="RGB",
                height=340,
            )
            with gr.Row():
                btn = gr.Button(" Analyze Image", variant="primary", scale=3)
                clear_btn = gr.ClearButton(
                    components=[input_view],
                    value="✕ Clear",
                    scale=1
                )
            gr.Markdown(
                "<small style='color:#94a3b8;'>Accepted formats: JPG, PNG, BMP. "
                "Minimum resolution: 168×168 px.</small>",
                elem_id="disclaimer"
            )

        with gr.Column(scale=1):
            gr.Markdown("### Grad-CAM Map")
            output_plot = gr.Image(
                label="Showing Grad-cam heatmaps to explain results",
                height=340,
                interactive=False,
            )
            gr.Markdown(
                "<small style='color:#94a3b8;'>Warmer regions (redish/yellow) indicates the regions "
                "that the model weighted most heavily in its preditiion.</small>",
                elem_id="disclaimer"
            )

    gr.Markdown("### Analysis Results!")
    output_html = gr.HTML(
        "<div style='padding:16px;color:#94a3b8;text-align:center;"
        "border:1px dashed #e2e8f0;border-radius:8px;'>"
        "Results will appear here after analysis is finished.</div>"
    )
    output_md = gr.Markdown(visible=False)

    gr.Markdown("---")

    btn.click(
        fn=predict_and_explain,
        inputs=input_view,
        outputs=[output_plot, output_html, output_md]
    )

if __name__ == "__main__":
    demo.launch()
