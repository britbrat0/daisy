import streamlit as st
from PIL import Image
import os
import io
import replicate

# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(page_title="Virtual Try-On", layout="wide")
st.title("👗 Virtual Clothing Try-On")

# -------------------------------------------------
# Paths
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "assets", "models")

# -------------------------------------------------
# Load stock models
# -------------------------------------------------
stock_models = {
    f.replace(".jpg", ""): os.path.join(MODELS_DIR, f)
    for f in os.listdir(MODELS_DIR)
    if f.endswith(".jpg")
}

model_choice = st.selectbox(
    "Choose a model",
    options=list(stock_models.keys())
)

model_img = Image.open(stock_models[model_choice]).convert("RGB")

# -------------------------------------------------
# Upload clothing
# -------------------------------------------------
clothing_file = st.file_uploader(
    "Upload Clothing Image",
    type=["png", "jpg", "jpeg"]
)

# -------------------------------------------------
# Display inputs
# -------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.subheader("Model")
    st.image(model_img, use_container_width=True)

with col2:
    st.subheader("Clothing")
    if clothing_file:
        clothing_img = Image.open(clothing_file).convert("RGBA")
        st.image(clothing_img, use_container_width=True)
    else:
        st.info("Upload clothing to continue")

# -------------------------------------------------
# Generate Try-On
# -------------------------------------------------
if clothing_file and st.button("Generate Try-On"):
    with st.spinner("Segmenting clothing..."):
        segmented = remove(clothing_img)

        buf = io.BytesIO()
        segmented.save(buf, format="PNG")
        buf.seek(0)

    with st.spinner("Running TryOnDiffusion..."):
        output = replicate.run(
            "yisol/tryondiffusion",
            input={
                "model_image": model_img,
                "garment_image": buf,
                "steps": 25,
                "guidance_scale": 2.5
            }
        )

        result_url = output[0]

    st.subheader("Result")
    st.image(result_url, use_container_width=True)
