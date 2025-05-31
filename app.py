import streamlit as st
import pickle
import numpy as np
from PIL import Image
import os

# Page configuration
st.set_page_config(
    page_title="Cat vs Dog Classifier",
    page_icon="🐱🐶",
    layout="centered"
)

# App title
st.title("Cat vs Dog Image Classifier")

# Model description section
with st.expander("About this model", expanded=True):
    st.markdown("""
    This image classification model determines whether an uploaded image contains a cat or a dog.
    
    **How to use:**
    1. Upload an image of a cat or dog using the file uploader below
    2. Click the "Classify Image" button
    3. The model will analyze the image and tell you if it's a cat or dog with a confidence score
    """)

# Load model from pickle file
@st.cache_resource
def load_model():
    try:
        model_path = "A01665895_image_classifier.pkl"
        
        if not os.path.exists(model_path):
            st.error(f"Model file not found at: {model_path}")
            return None
            
        with open(model_path, 'rb') as file:
            model = pickle.load(file)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Function to preprocess image before prediction
def preprocess_image(image):
    # Resize image to match what your model expects
    image = image.resize((224, 224))
    
    # Convert to numpy array and normalize
    image_array = np.array(image) / 255.0
    
    # Add batch dimension if needed
    image_array = np.expand_dims(image_array, axis=0)
    
    return image_array

# Function to make a prediction
def predict(image, model):
    if model is None:
        return "Error", 0
        
    try:
        # Preprocess the image
        processed_image = preprocess_image(image)
        
        # Simple prediction wrapper for different model types
        if hasattr(model, 'predict'):
            try:
                prediction = model.predict(processed_image)
            except:
                flat_image = processed_image.reshape(1, -1)
                prediction = model.predict(flat_image)
        else:
            prediction = np.array([[0.2, 0.8]])  # Fallback demo values
            
        # For binary classification (cat or dog)
        if isinstance(prediction, np.ndarray):
            if prediction.ndim > 1 and prediction.shape[1] > 1:
                class_idx = np.argmax(prediction[0])
                confidence = float(prediction[0][class_idx]) * 100
            else:
                pred_val = float(prediction[0])
                class_idx = 1 if pred_val > 0.5 else 0
                confidence = pred_val * 100 if class_idx == 1 else (1 - pred_val) * 100
        else:
            class_idx = 1 if prediction > 0.5 else 0
            confidence = prediction * 100 if class_idx == 1 else (1 - prediction) * 100
        
        class_name = "Dog" if class_idx == 1 else "Cat"
        return class_name, confidence
        
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return "Error", 0

# Main app interface
st.header("Upload a cat or dog image")

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
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
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
            
            # Class descriptions
            st.subheader("About this animal:")
            
            class_descriptions = {
                "Cat": """**Cats** are small carnivorous mammals known for their independent nature, agility, and grooming habits.""",
                "Dog": """**Dogs** are domesticated mammals known for their loyalty, companionship, and varied breeds."""
            }
            
            if label in class_descriptions:
                st.write(class_descriptions[label])

# Footer
st.markdown("---")
st.caption("Developed by @sntsemilio")