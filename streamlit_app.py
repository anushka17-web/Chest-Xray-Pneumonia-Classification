
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from pathlib import Path
from tensorflow.keras import layers

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Chest X-ray AI Analysis",
    page_icon="🩺",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "results" / "models"

IMG_SIZE = (224, 224)

MODEL_NAMES = [
    "ResNet50",
    "EfficientNetV2B0",
    "ConvNeXtTiny"
]


# --------------------------------------------------
# PREPROCESSING FUNCTIONS
# --------------------------------------------------

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


# --------------------------------------------------
# LOAD TRAINED MODELS
# --------------------------------------------------

@st.cache_resource(show_spinner="Loading trained AI models...")
def load_models():

    loaded_models = {}

    for model_name in MODEL_NAMES:

        backbone_path = MODEL_DIR / f"{model_name}_backbone.keras"
        head_path = MODEL_DIR / f"{model_name}_head.keras"

        base_model = tf.keras.models.load_model(
            backbone_path,
            compile=False
        )

        head = tf.keras.models.load_model(
            head_path,
            compile=False
        )

        pooled = layers.GlobalAveragePooling2D(
            name=f"{model_name}_global_average_pooling"
        )(base_model.output)

        feature_extractor = tf.keras.Model(
            inputs=base_model.input,
            outputs=pooled
        )

        loaded_models[model_name] = {
            "feature_extractor": feature_extractor,
            "head": head,
            "preprocess": preprocess_functions[model_name]
        }

    return loaded_models


# --------------------------------------------------
# PREDICTION FUNCTION
# --------------------------------------------------

def predict(image, models):

    image = image.convert("RGB").resize(IMG_SIZE)

    image_array = np.array(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    results = []

    for model_name, components in models.items():

        processed = components["preprocess"](
            tf.convert_to_tensor(image_array)
        )

        features = components["feature_extractor"](
            processed,
            training=False
        )

        probability = components["head"].predict(
            features,
            verbose=0
        ).reshape(-1)[0]

        probability = float(probability)

        label = (
            "PNEUMONIA"
            if probability >= 0.5
            else "NORMAL"
        )

        score = probability * 100

        color = (
            "#dc2626"
            if label == "PNEUMONIA"
            else "#16a34a"
        )

        results.append({
            "name": model_name,
            "label": label,
            "score": score,
            "color": color
        })

    return results


# --------------------------------------------------
# DASHBOARD DESIGN
# --------------------------------------------------

st.markdown("""
<style>

.main {
    background-color: #f8fafc;
}

.block-container {
    padding-top: 2rem;
}

.dashboard-title {
    text-align: center;
    color: #1e293b;
    font-size: 36px;
    font-weight: 700;
    margin-bottom: 5px;
}

.dashboard-subtitle {
    text-align: center;
    color: #64748b;
    font-size: 17px;
    margin-bottom: 30px;
}

.model-card {
    background: white;
    border: 1px solid #e2e8f0;
    border-radius: 16px;
    padding: 22px;
    min-height: 185px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.06);
}

.model-name {
    color: #334155;
    font-size: 20px;
    font-weight: 600;
    margin-top: 0;
}

.prediction-label {
    font-size: 25px;
    font-weight: 700;
    margin-bottom: 10px;
}

.score-text {
    color: #64748b;
    font-size: 14px;
    margin-bottom: 10px;
}

.progress-background {
    background: #e2e8f0;
    height: 10px;
    border-radius: 10px;
    overflow: hidden;
}

.progress-fill {
    height: 10px;
    border-radius: 10px;
}

</style>
""", unsafe_allow_html=True)


# --------------------------------------------------
# DASHBOARD HEADER
# --------------------------------------------------

st.markdown("""
<div class="dashboard-title">
    🩺 Chest X-ray AI Analysis
</div>

<div class="dashboard-subtitle">
    Pneumonia Classification Dashboard
</div>
""", unsafe_allow_html=True)

st.markdown("""
Upload a chest X-ray to compare predictions from three deep learning models.
""")

st.divider()


# --------------------------------------------------
# IMAGE UPLOAD
# --------------------------------------------------

left_column, right_column = st.columns([1, 2])

with left_column:

    st.subheader("Upload Chest X-ray")

    uploaded_file = st.file_uploader(
        "Choose an X-ray image",
        type=["png", "jpg", "jpeg"]
    )

    if uploaded_file is not None:

        image = Image.open(uploaded_file).convert("RGB")

        st.image(
            image,
            caption="Uploaded Chest X-ray",
            use_container_width=True
        )

    analyze_button = st.button(
        "🔍 Analyze X-ray",
        type="primary",
        use_container_width=True
    )


# --------------------------------------------------
# MODEL COMPARISON
# --------------------------------------------------

with right_column:

    st.subheader("AI Model Comparison")

    if analyze_button:

        if uploaded_file is None:

            st.warning("Please upload a chest X-ray image first.")

        else:

            try:

                with st.spinner("Analyzing X-ray using all three models..."):

                    models = load_models()
                    predictions = predict(image, models)

                result_columns = st.columns(3)

                for column, result in zip(
                    result_columns,
                    predictions
                ):

                    score = result["score"]
                    color = result["color"]

                    card_html = f"""
                    <div class="model-card">

                        <div class="model-name">
                            {result["name"]}
                        </div>

                        <div class="prediction-label"
                             style="color:{color};">
                            {result["label"]}
                        </div>

                        <div class="score-text">
                            Pneumonia model score: {score:.2f}%
                        </div>

                        <div class="progress-background">
                            <div class="progress-fill"
                                 style="width:{score:.2f}%;
                                        background:{color};">
                            </div>
                        </div>

                    </div>
                    """

                    with column:
                        st.html(card_html)

            except Exception as error:

                st.error(f"Prediction failed: {error}")

    else:

        st.info(
            "Upload a chest X-ray image and click "
            "'Analyze X-ray' to see model predictions."
        )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.markdown("""
**Models:** ResNet50 · EfficientNetV2B0 · ConvNeXtTiny

<small>Educational prototype only. This tool is not intended
for medical diagnosis or clinical decision-making.</small>
""", unsafe_allow_html=True)
