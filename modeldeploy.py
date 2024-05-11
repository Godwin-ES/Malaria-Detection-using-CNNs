import streamlit as st
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from PIL import Image
import tensorflow as tf
import time
import joblib

# Load the models
classify = joblib.load('classify.pkl')
model = joblib.load('model.pkl')

def main():
    st.set_page_config(page_title="Malaria Detector", page_icon="⚕️", layout="centered", initial_sidebar_state="auto")
    st.title('Blood Cell Image Analysis for Malaria Detection')
    st.write("Please upload an image of a blood cell.")

    upload = st.file_uploader("Upload cell image", type=["jpg", "png", "jpeg"])

    if st.button('Analyse'):
        if upload is None:
            st.error("Please upload an image.")
            return

        st.success("Image successfully uploaded!")

        # Display uploaded image
        st.image(upload, caption='Uploaded Image.', use_column_width=True)

        # Convert the uploaded file to a numpy array
        pil_image = Image.open(upload)

        # Preprocess the image (resize, normalize, etc.)
        resized_image = pil_image.resize((128, 128))
        normalized_image = np.array(resized_image) / 255.0
        img = np.expand_dims(normalized_image, axis=0)

        # Make prediction using classify_model to determine if it's a cell image
        cell_y = classify.predict(img)
        cell_prob = tf.nn.sigmoid(cell_y).numpy()
        
        # Check if the uploaded image is a cell image
        if cell_prob >= 0.5:
            st.write("Uploaded image is a cell image. Analyzing for malaria...")
            # Make prediction using malaria_model
            with st.spinner('Analyzing image for malaria...'):
                start_time = time.time()
                yout = model.predict(img)
                yprob = tf.nn.sigmoid(yout).numpy()
                end_time = time.time()

            # Determine the result and display prediction confidence
            if yprob >= 0.5:
                confidence = f"Confidence: {yprob[0][0]:.2f}"
                st.write(f"Prediction: Infected ({confidence})")
            else:
                confidence = f"Confidence: {1 - yprob[0][0]:.2f}"
                st.write(f"Prediction: Uninfected ({confidence})")
            
            # Show time taken for analysis
            analysis_time = end_time - start_time
            st.write(f"Time taken for analysis: {analysis_time:.2f} seconds")

            # Add a footer with model information
            st.markdown("Model: Convolutional Neural Network (CNN) trained for Malaria Detection")
        else:
            st.error("Uploaded image is not a cell image. Please upload a valid cell image.")

    # Keep the Streamlit app running
    st.stop()

if __name__ == '__main__':
    main()