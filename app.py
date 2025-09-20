import streamlit as st
import numpy as np
from PIL import Image
from models import predict
from utils import IMAGE_SIZE

st.set_page_config(page_title="Cat or Not", page_icon="🐱")

st.title("🐱 Cat or Not — Upload an image")

# Load saved parameters
params = np.load("model_params.npz")
w = params["w"]
b = float(params["b"])
image_size = int(params.get("image_size", IMAGE_SIZE))

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded image", use_container_width=True)

    # Preprocess
    img_resized = image.resize((image_size, image_size))
    img_arr = np.asarray(img_resized).reshape(-1, 1) / 255.0  # shape (num_features, 1)

    # Get probability and decision
    prob = predict(w, b, img_arr)[0, 0]
    is_cat = prob > 0.5

    st.write(f"Probability cat: {prob:.3f}")
    if is_cat:
        st.success("✅ It's likely a cat!")
    else:
        st.error("❌ Not a cat (or uncertain).")
    st.write("Note: Model may not be accurate for all images.")