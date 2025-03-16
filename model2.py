import streamlit as st
import requests
import tensorflow as tf
from PIL import Image
from io import BytesIO
import numpy as np

# Function to load the CNN model (assuming you have a trained model)
def load_model():
    model = tf.keras.models.load_model('skin_care_model.h5')  # Replace with your model path
    return model

# Function to preprocess the uploaded skin image
def preprocess_image(image):
    # Resize and normalize the image to match your CNN model's input requirements
    image = image.resize((224, 224))  # Assuming 224x224 input size for CNN
    image_array = np.array(image) / 255.0  # Normalize to [0, 1]
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Function to predict skin issue using the CNN model
def predict_skin_issue(model, image_array):
    predictions = model.predict(image_array)
    # Assuming you have a class index mapping (e.g., [0: Acne, 1: Dark Spots, 2: Wrinkles, ...])
    class_index = np.argmax(predictions, axis=1)[0]
    
    # Map the class index to a skin problem
    skin_problems = ['Acne', 'Dark Spots', 'Fine Lines and Wrinkles', 'Dry Skin']
    skin_problem = skin_problems[class_index]
    return skin_problem

# Function to get remedies from the Gemini API
def get_remedies_from_gemini(skin_problem, user_image):
    api_url = 'https://api.gemini.com/v1/analyze_skin'
    api_key = 'AIzaSyDxib8bIk3gqykdMGM6BAvujCeriidin64'  # Replace with your Gemini API key
    
    files = {'file': open(user_image, 'rb')}
    headers = {'Authorization': f'Bearer {api_key}'}
    
    response = requests.post(api_url, files=files, headers=headers)
    
    if response.status_code == 200:
        result = response.json()
        # Process the response and return the remedies
        return result.get('remedies', 'No remedies found based on the analysis.')
    else:
        return 'Error: Unable to fetch analysis from Gemini API.'

# Main function to display skin care remedies
def display_skin_care_remedies(user_image):
    model = load_model()
    
    # Preprocess the uploaded skin image
    image = Image.open(user_image)
    image_array = preprocess_image(image)
    
    # Predict the skin issue using the CNN model
    skin_problem = predict_skin_issue(model, image_array)
    
    # Display the predicted skin problem
    st.write(f"Predicted Skin Problem: {skin_problem}")
    
    # Get remedies based on the predicted skin issue using Gemini API
    remedies = get_remedies_from_gemini(skin_problem, user_image)
    
    # Display the remedies received from Gemini API
    st.write(f"Recommended Remedies for {skin_problem}:")
    st.write(remedies)

# Streamlit interface for user interaction
st.title("Skin Care Recommendation System")

user_image = st.file_uploader("Upload Your Skin Image", type=["jpg", "png", "jpeg"])

if user_image:
    display_skin_care_remedies(user_image)
