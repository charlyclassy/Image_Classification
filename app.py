

import streamlit as st
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from tensorflow.keras.applications.efficientnet_v2 import (
    EfficientNetV2B0,
    decode_predictions
)

# =====================================
# PAGE CONFIGURATION
# =====================================

st.set_page_config(
    page_title="AI Image Classifier",
    page_icon="🧠",
    layout="wide"
)

# =====================================
# CUSTOM CSS DESIGN
# =====================================

st.markdown("""
<style>
.stApp {
    background-color: #0f172a;
    color: #f8fafc;
}

.block-container {
    max-width: 1100px;
    padding-top: 2rem;
}

.hero {
    background: linear-gradient(135deg, #1e293b, #312e81);
    border-radius: 18px;
    padding: 40px 20px;
    text-align: center;
    margin-bottom: 30px;
}

.hero h1 {
    color: white;
    font-size: 42px;
}

.hero p {
    color: #cbd5e1;
    font-size: 17px;
}

.stButton > button {
    background-color: #6366f1;
    color: white;
    border-radius: 10px;
    padding: 10px 24px;
    font-weight: bold;
    border: none;
}

.stButton > button:hover {
    background-color: #4f46e5;
    color: white;
}

[data-testid="stFileUploader"] {
    border: 1px solid #475569;
    border-radius: 12px;
    padding: 12px;
}

</style>
""", unsafe_allow_html=True)

# =====================================
# APPLICATION HEADER
# =====================================

st.markdown("""
<div class="hero">
    <h1>🧠 AI Image Classifier</h1>
    <p>
        Intelligent Image Recognition Using
        EfficientNetV2 Deep Learning
    </p>
</div>
""", unsafe_allow_html=True)

# =====================================
# SIDEBAR
# =====================================

with st.sidebar:

    st.title("🔍 About This Project")

    st.write(
        "An AI-powered image classification "
        "application built with Python, "
        "TensorFlow and Streamlit."
    )

    st.divider()

    st.subheader("🤖 Model Information")

    st.write("**Architecture:** EfficientNetV2B0")
    st.write("**Type:** Convolutional Neural Network")
    st.write("**Dataset:** ImageNet")
    st.write("**Input Size:** 224 × 224")
    st.write("**Categories:** 1,000")

    st.divider()

    st.subheader("✨ Features")

    st.write("✅ AI Image Recognition")
    st.write("✅ CNN Deep Learning")
    st.write("✅ Top 3 Predictions")
    st.write("✅ Confidence Scores")
    st.write("✅ Image Preprocessing")

    st.divider()

    st.caption("Built with TensorFlow and Streamlit")

# =====================================
# LOAD EFFICIENTNET MODEL
# =====================================

@st.cache_resource
def load_model():

    model = EfficientNetV2B0(
        weights="imagenet",
        include_top=True,
        include_preprocessing=True
    )

    return model

# =====================================
# IMAGE PREPROCESSING
# =====================================

def preprocess_image(image):

    # Fix image orientation
    image = ImageOps.exif_transpose(image)

    # Convert to RGB
    image = image.convert("RGB")

    # Resize while preserving proportions
    image = ImageOps.pad(
        image,
        (224, 224),
        method=Image.Resampling.BICUBIC,
        color=(0, 0, 0)
    )

    # Convert image to NumPy array
    image_array = np.asarray(
        image,
        dtype=np.float32
    )

    # Add batch dimension
    image_array = np.expand_dims(
        image_array,
        axis=0
    )

    # EfficientNetV2 includes preprocessing
    # internally and expects values 0-255.

    return image_array

# =====================================
# IMAGE UPLOAD SECTION
# =====================================

st.subheader("📤 Upload Your Image")

st.write(
    "Upload a JPG, JPEG or PNG image "
    "and let AI identify the object."
)

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

# =====================================
# CLASSIFICATION SECTION
# =====================================

if uploaded_file is not None:

    try:

        image = Image.open(uploaded_file)
        image = ImageOps.exif_transpose(image)
        image = image.convert("RGB")

        col1, col2 = st.columns(
            2,
            gap="large"
        )

        with col1:

            st.subheader("🖼️ Image Preview")

            st.image(
                image,
                caption="Uploaded Image",
                use_container_width=True
            )

        with col2:

            st.subheader("🤖 AI Analysis")

            st.success(
                "Image uploaded successfully!"
            )

            classify = st.button(
                "🚀 Classify Image",
                type="primary",
                use_container_width=True
            )

            if classify:

                try:

                    with st.spinner(
                        "AI is analysing your image..."
                    ):

                        # Load CNN model
                        model = load_model()

                        # Prepare input
                        processed_image = preprocess_image(
                            image
                        )

                        # Generate predictions
                        predictions = model.predict(
                            processed_image,
                            verbose=0
                        )

                        # Decode ImageNet predictions
                        results = decode_predictions(
                            predictions,
                            top=3
                        )[0]

                    st.success(
                        "Classification completed!"
                    )

                    st.subheader(
                        "📊 Prediction Results"
                    )

                    # Display top 3 predictions
                    for rank, (_, label, confidence) in enumerate(
                        results,
                        start=1
                    ):

                        clean_label = (
                            label.replace("_", " ").title()
                        )

                        st.write(
                            f"**{rank}. {clean_label}** "
                            f"— {confidence * 100:.2f}%"
                        )

                        st.progress(
                            float(confidence)
                        )

                    # Highest prediction
                    best_label = (
                        results[0][1]
                        .replace("_", " ")
                        .title()
                    )

                    best_confidence = (
                        float(results[0][2]) * 100
                    )

                    st.info(
                        f"🏆 Top Prediction: {best_label}"
                    )

                    st.metric(
                        "Top Prediction Confidence",
                        f"{best_confidence:.2f}%"
                    )

                    if best_confidence < 40:
                        st.warning(
                            "The model has low confidence "
                            "in its top prediction. "
                            "Try a clearer image."
                        )

                    st.caption(
                        "Predictions come from 1,000 "
                        "ImageNet categories. Confidence "
                        "does not guarantee correctness."
                    )

                except Exception as error:

                    st.error(
                        "The model could not complete "
                        "classification."
                    )

                    st.exception(error)

    except (
        UnidentifiedImageError,
        OSError,
        ValueError
    ):

        st.error(
            "Invalid image. Please upload "
            "a valid JPG or PNG file."
        )

else:

    st.info(
        "👆 Upload an image above to get started."
    )

# =====================================
# FOOTER
# =====================================

st.divider()

st.markdown(
    """
    <p style="
        text-align:center;
        color:#94a3b8;
        font-size:14px;
    ">
        AI Image Classification |
        EfficientNetV2 CNN |
        Built with Streamlit
    </p>
    """,
    unsafe_allow_html=True
)
