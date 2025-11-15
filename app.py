import streamlit as st
from PIL import Image
import torch
from diffusers import StableDiffusionPipeline

# ---------------------
# Page config
# ---------------------
st.set_page_config(page_title="Virtual Try-On", layout="wide")

st.title("Virtual Clothing Try-On")
st.write("Upload a clothing image and see it on a model.")

# ---------------------
# File upload
# ---------------------
clothing_file = st.file_uploader("Upload Clothing Image", type=["png", "jpg", "jpeg"])
model_file = st.file_uploader("Upload Model Image (optional)", type=["png", "jpg", "jpeg"])

# Load uploaded or stock model image
if model_file:
    model_img = Image.open(model_file).convert("RGB")
else:
    model_img = Image.open("stock_model.jpg").convert("RGB")

if clothing_file:
    clothing_img = Image.open(clothing_file).convert("RGBA")
    st.image(clothing_img, caption="Clothing Uploaded", use_column_width=True)

st.image(model_img, caption="Model", use_column_width=True)

# ---------------------
# Inference button
# ---------------------
if clothing_file and st.button("Generate Try-On"):
    with st.spinner("Generating image..."):
        # Load your try-on model (replace with TryOnDiffusion later)
        pipe = StableDiffusionPipeline.from_pretrained(
            "CompVis/stable-diffusion-v1-4",
            torch_dtype=torch.float16
        ).to("cuda")

        # Simple prompt for demo
        prompt = "A person wearing the uploaded clothing on a stock model"

        # Convert clothing to RGB
        clothing_img_rgb = clothing_img.convert("RGB")

        # Generate image
        result = pipe(prompt=prompt, image=clothing_img_rgb, num_inference_steps=25).images[0]

        st.image(result, caption="Virtual Try-On Result", use_column_width=True)
