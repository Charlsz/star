from PIL import Image
import torch
import streamlit as st

from star.predict import load_model, predict_image


st.set_page_config(page_title="STAR", page_icon="🌌", layout="centered")

st.title("STAR")
st.subheader("Galaxy Morphology Classifier")
st.write(
    "Upload a galaxy image and the model will predict whether it is "
    "smooth or featured based on Galaxy Zoo-style morphology labels."
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = load_model(device)

uploaded_file = st.file_uploader(
    "Upload a galaxy image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    if st.button("Predict"):
        predicted_class, confidence, probabilities = predict_image(
            uploaded_file, model, device
        )

        if predicted_class == "smooth":
            friendly_label = "Smooth / likely elliptical"
            explanation = (
                "The model thinks this galaxy looks smooth and rounded, "
                "with no obvious visible disk features."
            )
        else:
            friendly_label = "Featured / likely disk or spiral"
            explanation = (
                "The model thinks this galaxy shows visible structure, "
                "such as a disk, arms, or other features."
            )

        st.success(f"Prediction: {friendly_label}")
        st.write(f"Confidence: {confidence:.4f}")
        st.write(explanation)

        st.write("### Galaxy Zoo class probabilities")
        st.write(f"Smooth: {probabilities[0]:.4f}")
        st.write(f"Featured or disk: {probabilities[1]:.4f}")

st.markdown("---")
st.caption(
    "Note: This model is trained on Galaxy Zoo 2 smooth vs. features/disk labels, "
    "so the friendly descriptions are interpretations rather than exact class names."
)