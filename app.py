import streamlit as st
import numpy as np
from PIL import Image
from tensorflow import keras
import pickle

st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="🐱🐶",
    layout="centered"
)

st.title("Cat vs Dog Image Classifier")

st.markdown("""
The model processes your image and predicts whether it's a cat or a dog.
**How to use:**
1. Upload an image.
2. Click "Classify Image".
3. See the results!
""")

@st.cache_resource
def load_model():
    with open("A01665895_image_classifier.pkl", "rb") as file:
        model = pickle.load(file)
    return model

def preprocess_image(image):
    image = image.resize((96, 96))  # change from (224, 224)
    image_array = np.array(image)
    image_array = image_array / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array

def predict(image, model):
    image_array = preprocess_image(image)
    preds = model.predict(image_array)
    class_idx = preds.argmax(axis=1)[0]
    confidence = float(preds[0][class_idx]) * 100
    class_name = "Dog" if class_idx == 1 else "Cat"
    return class_name, confidence

st.header("Upload an image")
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    if st.button("Classify Image"):
        with st.spinner("Classifying..."):
            model = load_model()
            label, confidence = predict(image, model)
            st.subheader("Classification Results:")
            st.write(f"**Predicted Class:** {label}")
            st.write(f"**Confidence:** {confidence:.2f}%")
            st.progress(min(confidence/100, 1.0))
            st.subheader("About this animal:")
            st.write("**Cats** are small carnivorous mammals known for their independent nature." if label == "Cat"
                     else "**Dogs** are domesticated mammals known for their loyalty and companionship.")

st.markdown("---")
st.caption("Developed by @sntsemilio")