
import gradio as gr
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path
from tensorflow.keras import layers

# Project configuration
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "results" / "models"

IMG_SIZE = (224, 224)
MODEL_NAMES = ["ResNet50", "EfficientNetV2B0", "ConvNeXtTiny"]

# Preprocessing functions
def resnet_preprocess(images):
    return tf.keras.applications.resnet50.preprocess_input(
        tf.cast(images, tf.float32)
    )

def efficientnet_preprocess(images):
    return tf.cast(images, tf.float32)

def convnext_preprocess(images):
    return tf.cast(images, tf.float32)

preprocess_functions = {
    "ResNet50": resnet_preprocess,
    "EfficientNetV2B0": efficientnet_preprocess,
    "ConvNeXtTiny": convnext_preprocess,
}

# Load trained model components
models = {}

print("Loading trained models...")

for model_name in MODEL_NAMES:
    backbone_path = MODEL_DIR / f"{model_name}_backbone.keras"
    head_path = MODEL_DIR / f"{model_name}_head.keras"

    base_model = tf.keras.models.load_model(
        backbone_path, compile=False
    )

    head = tf.keras.models.load_model(
        head_path, compile=False
    )

    pooled = layers.GlobalAveragePooling2D(
        name=f"{model_name}_global_average_pooling"
    )(base_model.output)

    feature_extractor = tf.keras.Model(
        inputs=base_model.input,
        outputs=pooled
    )

    models[model_name] = {
        "feature_extractor": feature_extractor,
        "head": head,
        "preprocess": preprocess_functions[model_name]
    }

    print(model_name, "loaded successfully")

# Prediction function

def predict(image):
    if image is None:
        message = """
        <div style="padding:20px; border-radius:12px;
                    background:#f1f5f9; text-align:center;">
            Please upload a chest X-ray image.
        </div>
        """
        return message, message, message

    image = image.convert("RGB").resize(IMG_SIZE)

    image_array = np.array(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    results = []

    for model_name, components in models.items():
        processed = components["preprocess"](
            tf.convert_to_tensor(image_array)
        )

        features = components["feature_extractor"](
            processed, training=False
        )

        probability = components["head"].predict(
            features, verbose=0
        ).reshape(-1)[0]

        label = (
            "PNEUMONIA" if probability >= 0.5 else "NORMAL"
        )

        score = probability * 100

        color = "#dc2626" if label == "PNEUMONIA" else "#16a34a"

        card = f"""
        <div style="
            background:white;
            border:1px solid #e2e8f0;
            border-radius:16px;
            padding:22px;
            min-height:180px;
            box-shadow:0 4px 12px rgba(0,0,0,0.06);
        ">
            <h3 style="margin-top:0; color:#334155;">
                {model_name}
            </h3>

            <h2 style="color:{color}; margin-bottom:8px;">
                {label}
            </h2>

            <p style="color:#64748b; margin-bottom:8px;">
                Pneumonia model score: {score:.2f}%
            </p>

            <div style="
                background:#e2e8f0;
                height:10px;
                border-radius:10px;
                overflow:hidden;
            ">
                <div style="
                    width:{score:.2f}%;
                    background:{color};
                    height:10px;
                    border-radius:10px;
                "></div>
            </div>
        </div>
        """

        results.append(card)

    return tuple(results)

# Gradio interface

# Modern Medical Dashboard
with gr.Blocks(theme=gr.themes.Soft(
    primary_hue="blue",
    secondary_hue="sky",
    neutral_hue="slate"
)) as demo:

    gr.Markdown(
        """
        # 🩺 Chest X-ray AI Analysis
        ### Pneumonia Classification Dashboard

        Upload a chest X-ray to compare predictions from three deep learning models.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                type="pil",
                label="Upload Chest X-ray",
                height=350
            )

            predict_button = gr.Button(
                "Analyze X-ray",
                variant="primary"
            )

        with gr.Column(scale=2):
            gr.Markdown("### AI Model Comparison")

            with gr.Row():
                output_resnet = gr.HTML()
                output_efficientnet = gr.HTML()
                output_convnext = gr.HTML()

    gr.Markdown(
        """
        ---
        **Models:** ResNet50 · EfficientNetV2B0 · ConvNeXtTiny

        <small>Educational prototype only. This tool is not intended
        for medical diagnosis or clinical decision-making.</small>
        """
    )

    predict_button.click(
        fn=predict,
        inputs=image_input,
        outputs=[
            output_resnet,
            output_efficientnet,
            output_convnext
    ]
)

if __name__ == "__main__":
    demo.launch()
