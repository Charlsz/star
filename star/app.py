import streamlit as st
from PIL import Image

from star.predict import predict_image

st.set_page_config(page_title="STAR", page_icon="🌌", layout="centered")

st.title("STAR")
st.subheader("Space Taxonomy and Analysis Recognition")
st.write("Galaxy morphology classifier - first version")

uploaded_file = st.file_uploader(
    "Upload a galaxy image",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    result = predict_image(image)

    st.markdown(f"### Prediction: {result['label']}")
    st.write(f"Confidence: {result['confidence']:.4f}")

    st.markdown("### Class probabilities")
    st.json(result["probabilities"])