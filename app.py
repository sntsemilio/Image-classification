import streamlit as st
import pickle
import numpy as np
from PIL import Image
import os

# Page configuration
st.set_page_config(
    page_title="Image Classification Model",
    page_icon="🖼️",
    layout="centered"
)

# App title
st.title("Image Classification Model")

# Model description section
with st.expander("About this model", expanded=True):
    st.markdown("""
    This image classification model was trained to categorize images into specific classes.
    
    The model uses [describe your model architecture here] and was trained on 
    [describe your dataset here]. It can classify images into [number of classes] categories.
    
    **How to use:**
    1. Upload an image using the file uploader below
    2. The model will process your image and provide a classification
    3. Results will show the predicted class and confidence score
    """)

# Load model from pickle file
@st.cache_resource
def load_model():
    try:
        model_path = "A01665895_Deep_Learning_image_classifier.pkl"  # Update this path to your actual model file
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Function to preprocess image before prediction
def preprocess_image(image):
    # Resize image to match what your model expects
    # Modify these parameters to match your model's input requirements
    image = image.resize((224, 224))
    
    # Convert to numpy array and normalize
    image_array = np.array(image) / 255.0
    
    # Add batch dimension if needed
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array

# Function to make a prediction
def predict(image, model):
    try:
        # Preprocess the image
        processed_image = preprocess_image(image)
        
        # Make prediction
        prediction = model.predict(processed_image)
        
        # Get predicted class
        # Modify this section based on your model's output format
        if hasattr(model, 'classes_'):
            class_idx = np.argmax(prediction, axis=1)[0]
            class_name = model.classes_[class_idx]
            confidence = float(prediction[0][class_idx]) * 100
        else:
            # If model doesn't have classes_ attribute
            class_idx = np.argmax(prediction)
            class_name = f"Class {class_idx}"
            confidence = float(prediction.flatten()[class_idx]) * 100
            
        return class_name, confidence
        
    except Exception as e:
        st.error(f"Prediction error: {e}")
        return "Error", 0

# Main app interface
st.header("Upload an image for classification")

# File uploader
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

# Load model
model = load_model()

if model is None:
    st.error("Failed to load the model. Please check if the model file exists in the correct location.")
else:
    st.success("Model loaded successfully!")

# When file is uploaded
if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Add a predict button
    if st.button("Classify Image"):
        with st.spinner("Classifying..."):
            # Make prediction
            label, confidence = predict(image, model)
            
            # Display results
            st.subheader("Classification Results:")
            st.write(f"**Predicted Class:** {label}")
            st.write(f"**Confidence:** {confidence:.2f}%")
            
            # Progress bar visualization
            st.progress(min(confidence/100, 1.0))
            
            # You can add class-specific descriptions here
            st.subheader("Class Description:")
            
            # Replace this dictionary with descriptions for your classes
            class_descriptions = {
                # "class_name": "Description of what this class represents",
                # Add more class descriptions here
            }
            
            if label in class_descriptions:
                st.write(class_descriptions[label])
            else:
                st.write("This class represents [add your description here].")

# Footer
st.markdown("---")
st.caption("Developed by @sntsemilio")