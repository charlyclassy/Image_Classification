
import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions
)

st.title("Deep Learning Image Classification")
st.write("Upload an image and let the CNN model identify it.")

@st.cache_resource
def load_model():
    return MobileNetV2(weights="imagenet")

model = load_model()

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image")

    if st.button("Classify Image"):
        img = image.resize((224, 224))
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(
            img_array.astype("float32")
        )

        predictions = model.predict(img_array)
        results = decode_predictions(predictions, top=3)[0]

        st.subheader("Classification Results")

        for _, label, confidence in results:
            st.write(f"{label}: {confidence * 100:.2f}%")
            st.progress(float(confidence))
