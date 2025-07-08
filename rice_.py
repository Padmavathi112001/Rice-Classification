import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image

# Load your trained rice classification model
model = load_model("rice_model.h5")  # Make sure this file is in the same folder

# Class names (update according to your model training)
class_names = ['Basmati', 'Jasmine', 'Arborio', 'Ipsala', 'Karacadag']

st.title("🍚 Rice Type Classifier")
st.write("Upload a rice grain image to classify its type.")

# File uploader
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).resize((224, 224))
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocessing
    img_array = img_to_array(image)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalization

    # Prediction
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions)

    st.success(f"Predicted: {predicted_class}")
    st.info(f"Confidence: {confidence:.2f}")
