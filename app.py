import streamlit as st
from PIL import Image
import pickle
import numpy as np
from skimage.transform import resize


categories = ['Organic', 'Recycle']
with open("model.p", 'rb') as f:
    model = pickle.load(f)


st.title("Waste Classification App")
st.write("Upload an image of waste to classify its type.")
uploaded_file = st.file_uploader("Upload the waste Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")
    if st.button("Classify"):
        image = np.array(image)
        image = resize(image, (50, 50, 3))
        image = np.expand_dims(image, axis=0)
        prediction = model.predict(image)
        print(prediction[0])
        label = list(prediction[0]).index(max(prediction[0]))
        st.write(f"This is a {categories[label]} Waste")
