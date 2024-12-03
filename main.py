import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st

# Set working directory and paths
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(working_dir, "trained_model/crop_disease_prediction_model.h5")
class_indices_path = os.path.join(working_dir, "class_indices.json")

# Load the pre-trained model and class indices once
@st.cache_resource
def load_model_and_indices():
    try:
        model = tf.keras.models.load_model(model_path)
        with open(class_indices_path, "r") as file:
            class_indices = json.load(file)
        return model, class_indices
    except Exception as e:
        st.error(f"Failed to load model or class indices: {e}")
        return None, None

model, class_indices = load_model_and_indices()

# Function to load and preprocess the image
def load_and_preprocess_image(image_file, target_size=(224, 224)):
    try:
        img = Image.open(image_file)
        if img.format not in ['JPEG', 'JPG', 'PNG']:
            raise ValueError("Unsupported image format. Please upload a JPG or PNG file.")
        img = img.resize(target_size)
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array.astype('float32') / 255.0
        return img_array
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

# Function to predict the class of an image
def predict_image_class(model, image_file, class_indices):
    if not model or not class_indices:
        st.error("Model or class indices are not properly loaded.")
        return None, None
    img_array = load_and_preprocess_image(image_file)
    if img_array is None:
        return None, None
    try:
        predictions = model.predict(img_array)
        st.write("Raw predictions:", predictions)  # Debugging: Display raw predictions
        predicted_class_index = np.argmax(predictions, axis=1)[0]
        predicted_class_name = class_indices.get(str(predicted_class_index), "Unknown")
        confidence = predictions[0][predicted_class_index] * 100
        return predicted_class_name, confidence
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return None, None

# Streamlit app interface
st.title('🌱Crop Disease Prediction')
st.markdown("")

# Upload image section
uploaded_image = st.file_uploader("Upload an image of the crop:", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    try:
        # Display uploaded image
        col1, col2 = st.columns(2)
        with col1:
            image = Image.open(uploaded_image)
            resized_img = image.resize((150, 150))
            st.image(resized_img, caption="Uploaded Image", use_container_width=True)

        # Perform prediction
        with col2:
            if st.button('Classify', key='classify_btn'):
                with st.spinner('Classifying...'):
                    prediction, confidence = predict_image_class(model, uploaded_image, class_indices)
                if prediction:
                    st.success(f"Prediction: {prediction}")
                    st.info(f"Confidence: {confidence:.2f}%")
                else:
                    st.error("Failed to classify the image.")
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Footer or additional instructions
st.markdown("""
### Instructions:
1. Upload a clear image of the crop.
2. Click the **Classify** button to predict the disease.
""")
