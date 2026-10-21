
import streamlit as st
import numpy as np
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions
)

# Page configuration
st.set_page_config(
    page_title="AI Image Classifier",
    page_icon="🧠",
    layout="wide"
)

# Custom styling
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
    text-align: center;
    padding: 32px 16px;
    background: linear-gradient(135deg, #1e293b, #312e81);
    border-radius: 18px;
    margin-bottom: 28px;
}

.hero h1 {
    color: #ffffff;
    font-size: 42px;
    margin-bottom: 10px;
}

.hero p {
    color: #cbd5e1;
    font-size: 17px;
}

.stButton > button {
    background: #6366f1;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 10px 24px;
    font-weight: 600;
}

.stButton > button:hover {
    background: #4f46e5;
    color: white;
}

[data-testid="stFileUploader"] {
    border: 1px solid #475569;
    border-radius: 12px;
    padding: 12px;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="hero">
    <h1>🧠 AI Image Classifier</h1>
    <p>
        Discover what is inside your images using
        Convolutional Neural Networks.
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.title("🔍 About This Project")
    st.write(
        "An AI-powered image classification application "
        "built with Python, TensorFlow and Streamlit."
    )

    st.divider()

    st.subheader("Model Information")
    st.write("**Architecture:** MobileNetV2")
    st.write("**Model Type:** CNN")
    st.write("**Training Dataset:** ImageNet")
    st.write("**Input Size:** 224 × 224")
    st.write("**Classes:** 1,000")

    st.divider()
    st.caption("Powered by TensorFlow and Streamlit")

# Load the model once
@st.cache_resource
def load_model():
    return MobileNetV2(weights="imagenet")

# Main application
st.subheader("📤 Upload Your Image")
st.write("Choose a JPG, JPEG or PNG image to classify.")

uploaded_file = st.file_uploader(
    "Select an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns(2, gap="large")

        with col1:
            st.subheader("🖼️ Image Preview")
            st.image(
                image,
                caption="Your uploaded image",
                use_container_width=True
            )

        with col2:
            st.subheader("🤖 AI Analysis")
            st.success("Image uploaded successfully!")

            if st.button(
                "🚀 Classify Image",
                type="primary"
            ):
                with st.spinner("AI is analysing your image..."):

                    # Image preprocessing
                    img = image.resize((224, 224))
                    img_array = np.asarray(
                        img, dtype=np.float32
                    )
                    img_array = np.expand_dims(
                        img_array, axis=0
                    )
                    img_array = preprocess_input(img_array)

                    # CNN predictions
                    model = load_model()
                    predictions = model.predict(
                        img_array, verbose=0
                    )

                    results = decode_predictions(
                        predictions, top=3
                    )[0]

                st.success("Classification complete!")
                st.subheader("📊 Prediction Results")

                for rank, (_, label, confidence) in enumerate(
                    results, start=1
                ):
                    label = label.replace("_", " ").title()

                    st.write(
                        f"**{rank}. {label}** "
                        f"— {confidence * 100:.2f}%"
                    )

                    st.progress(float(confidence))

                best_label = results[0][1]
                best_label = best_label.replace(
                    "_", " "
                ).title()

                st.info(
                    f"🏆 Top Prediction: {best_label}"
                )

                st.caption(
                    "Confidence scores represent model "
                    "predictions, not guaranteed accuracy."
                )

    except (UnidentifiedImageError, OSError, ValueError):
        st.error(
            "The image could not be opened. "
            "Please upload a valid JPG or PNG file."
        )

    except Exception:
        st.error(
            "Classification failed. Please check "
            "your model installation and app logs."
        )

else:
    st.info("👆 Upload an image above to get started.")

# Footer
st.divider()
st.markdown(
    "<p style='text-align:center;color:#94a3b8;'>"
    "AI Image Classification | Built with Streamlit"
    "</p>",
    unsafe_allow_html=True
)

