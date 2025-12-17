import streamlit as st
from PIL import Image
import os
import replicate

# -------------------------------------------------
# Page config
# -------------------------------------------------
st.set_page_config(page_title="Virtual Try-On", layout="wide")
st.title("👗 Virtual Clothing Try-On")

st.write("Upload a clothing image and see it on a stock model.")

# -------------------------------------------------
# Paths
# -------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "assets", "models")

# -------------------------------------------------
# Load stock models
# -------------------------------------------------
if not os.path.exists(MODELS_DIR):
    st.error("❌ assets/models folder not found.")
    st.stop()

model_files = [
    f for f in os.listdir(MODELS_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

if not model_files:
    st.error("❌ No stock model images found in assets/models/")
    st.stop()

model_choice = st.selectbox("Choose a model", model_files)
model_path = os.path.join(MODELS_DIR, model_choice)
model_img = Image.open(model_path).convert("RGB")

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
        clothing_img = Image.open(clothing_file).convert("RGB")
        st.image(clothing_img, use_container_width=True)
    else:
        st.info("Upload a clothing image to continue")

# -------------------------------------------------
# Generate Try-On
# -------------------------------------------------
if clothing_file and st.button("Generate Try-On"):
    with st.spinner("Running TryOnDiffusion (this may take ~30–60 seconds)..."):
        try:
            output = replicate.run(
                "yisol/tryondiffusion",
                input={
                    "model_image": model_img,
                    "garment_image": clothing_img,
                    "steps": 25,
                    "guidance_scale": 2.5
                }
            )

            result_url = output[0]

            st.subheader("Result")
            st.image(result_url, use_container_width=True)

        except Exception as e:
            st.error("❌ Try-on generation failed.")
            st.exception(e)

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.markdown("---")
st.caption("Powered by TryOnDiffusion • Streamlit • Replicate")
