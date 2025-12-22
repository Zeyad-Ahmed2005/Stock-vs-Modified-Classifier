import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

# -----------------------------
# Load model once
# -----------------------------
@st.cache_resource
def load_cnn():
    return load_model("model_img.h5")

model = load_cnn()

CLASS_LABELS = {0: "Stock", 1: "Modified"}

# -----------------------------
# Image preprocessing
# -----------------------------
def preprocess_image(image, target_size=(224, 224)):
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    image = cv2.resize(image, target_size)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# -----------------------------
# UI
# -----------------------------
st.set_page_config(page_title="Stock vs Modified Classifier", layout="centered")

st.title("🚗 Stock vs Modified Car Classifier")
st.write("Upload a car image and get prediction confidence")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Predict"):
        img = preprocess_image(image)
        preds = model.predict(img)[0]

        class_id = np.argmax(preds)
        confidence = float(preds[class_id])

        st.success(f"Prediction: **{CLASS_LABELS[class_id]}**")
        st.progress(confidence)
        st.write(f"Confidence: **{confidence:.2%}**")
