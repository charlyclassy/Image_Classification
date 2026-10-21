import streamlit as st

st.title("Deep Learning Image Classification")

st.write("Upload an image to identify the object.")

uploaded_file = st.file_uploader(
    "Choose an image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    st.image(uploaded_file, caption="Uploaded Image")
    st.success("Image uploaded successfully!")