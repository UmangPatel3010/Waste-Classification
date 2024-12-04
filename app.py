import streamlit as st
from PIL import Image
import numpy as np
import pickle
import gdown
import os
import cv2
import tensorflow as tf

# Define categories
categories = ['Organic', 'Recycle']

url = "https://drive.google.com/file/d/1szLBFHmsyTDOCAFgo_YGecyAebmmsJJx/view?usp=sharing"
output = "model.p"  # Local filename to save the file as
if not os.path.exists("model.p"):
    gdown.download(url, output, fuzzy=True)

# Load the trained model
with open(output, 'rb') as f:
    model = pickle.load(f)

# Streamlit UI
st.set_page_config(page_title="Waste Classification", layout="wide", page_icon="./favicon.png")
st.write('<style>div.block-container{padding-top:2rem;}</style>', unsafe_allow_html=True)
st.title(":green[Waste Classification App]")

# Create two columns: one for the image, one for the button and result

uploaded_file = st.file_uploader("Upload the waste Image", type=["jpg", "jpeg", "png"])
col1, col2, _ = st.columns(3, gap="large")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    col1.image(image, caption="Uploaded Image", use_column_width="auto")

    if col2.button("Classify"):
        image = np.array(image)
        image = tf.image.resize(image, [50, 50])
        image = np.expand_dims(image, axis=0)  # Add batch dimension

        prediction = model.predict(image)

        label = list(prediction[0]).index(max(prediction[0]))

        s = f"<p style='font-size:30px;'>Prediction: <b style='color:{"#64f562" if label == 0 else "#706bff"}';>{categories[label]} waste</b></p>"
        col2.markdown(s, unsafe_allow_html=True)
