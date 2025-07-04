import gradio as gr
import os
from utils.infer_explain import classify_and_explain

# 📂 Load sample images
EXAMPLES_DIR = "examples"
example_paths = sorted([
    [os.path.join(EXAMPLES_DIR, f)]
    for f in os.listdir(EXAMPLES_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
])

with gr.Blocks() as demo:
    gr.Markdown("# 🐾 Animal Classifier with Explainability")
    gr.Markdown("Upload or select an image to view predictions and Captum-based visual explanations.")

    with gr.Row():
        with gr.Column(scale=6):
            image_input = gr.Image(label="Input Image", type="pil", height=300, value=None)
            clear_btn = gr.Button("Clear Image")

            clear_btn.click(
                fn=lambda: None,
                inputs=[],
                outputs=image_input
            )

            gr.Examples(
            examples=example_paths,
            inputs=image_input,
            label="Sample Images",
            examples_per_page=8,
            )
        with gr.Column(scale=4):
            label_output = gr.Label(label="Top Predictions")
            image_output = gr.Image(label="Attribution Heatmap", height=260)

    image_input.change(
        fn=classify_and_explain,
        inputs=image_input,
        outputs=[label_output, image_output]
    )

demo.launch(show_error=True)
