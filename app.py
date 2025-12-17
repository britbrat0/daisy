import streamlit as st
from PIL import Image
import os
import torch
from diffusers import StableDiffusionPipeline

# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(
    page_title="Virtual Clothing Try-On",
    layout="wide"
)

st.title("👗 Virtual Clothing Try-On")
st.write("Upload a clothing image and preview it on a stock model.")

# -------------------------------------------------
# Paths (Streamlit Cloud safe)
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ASSETS_DIR = os.path.join(BASE_DIR, "assets")
STOCK_MODEL_PATH = os.path.join(ASSETS_DIR, "stock_model.jpg")

# -------------------------------------------------
# Sidebar
# -------------------------------------------------
st.sidebar.header("Options")
use_ai = st.sidebar.checkbox(
    "Generate AI Try-On (slow, CPU-only)",
    value=False,
    help="Enable AI image generation (very slow without GPU)"
)

# -------------------------------------------------
# File upload
# -------------------------------------------------
clothing_file = st.file_uploader(
    "Upload Clothing Image",
    type=["png", "jpg", "jpeg"]
)

model_file = st.file_uploader(
    "Upload Model Image (optional)",
    type=["png", "jpg", "jpeg"]
)

# -------------------------------------------------
# Load model image
# -------------------------------------------------
if model_file:
    model_img = Image.open(model_file).convert("RGB")
else:
    if not os.path.exists(STOCK_MODEL_PATH):
        st.error(
            "❌ stock_model.jpg not found.\n\n"
            "Make sure it exists at: assets/stock_model.jpg"
        )
        st.stop()
    model_img = Image.open(STOCK_MODEL_PATH).convert("RGB")

# -------------------------------------------------
# Display images
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
        st.info("Upload a clothing image to continue.")

# -------------------------------------------------
# Generate result
# -----------------------------------------------
