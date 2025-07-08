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
       
    
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42)
    model = SVC(kernel='linear', probability=True)
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test))
    return model, le, acc

model, label_encoder, acc = load_data_and_train()
# st.success(f"✅ Model trained with accuracy: {acc*100:.2f}%")

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
    st.write(f"📊 Confidence: **{prob*100:.2f}%**")
