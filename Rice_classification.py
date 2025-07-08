import streamlit as st
import os
import numpy as np
from PIL import Image
# Config
DATA_DIR = r"C:\Users\LENOVO\Downloads\Rice Classification"
IMG_SIZE = (64, 64)
st.title("🍚 Rice Classification (No TensorFlow)")
@st.cache_resource
def load_data_and_train():
    X, y = [], []
# Upload image
img_file = st.file_uploader("Upload rice image", type=["jpg", "jpeg", "png"])
if img_file:
    img = Image.open(img_file).resize(IMG_SIZE)
    img_np = np.array(img.convert("L"))  # Grayscale
    hist = cv2.calcHist([img_np], [0], None, [256], [0, 256]).flatten().reshape(1, -1)

    pred = model.predict(hist)[0]
    prob = model.predict_proba(hist)[0][pred]

    st.image(img, caption="Uploaded Image")
    st.write(f"🔍 Predicted: **{label_encoder.inverse_transform([pred])[0]}**")

