import streamlit as st
from tensorflow.keras.models import load_model
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
from PIL import Image
import tensorflow as tf
import time

model = load_model("my_model2.h5")

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

        # Make prediction
        with st.spinner('Analyzing image...'):
            start_time = time.time()
            yout = model.predict(img)
            yprob = tf.nn.sigmoid(yout).numpy()
            end_time = time.time()

        # Determine the result and display prediction confidence
        prediction = "Infected" if yprob >= 0.5 else "Uninfected"
        confidence = f"Confidence: {yprob[0][0]:.2f}"
        st.write(f"Prediction: {prediction} ({confidence})")
        
        # Show time taken for analysis
        analysis_time = end_time - start_time
        st.write(f"Time taken for analysis: {analysis_time:.2f} seconds")

        # Add a footer with model information
        st.markdown("Model: Convolutional Neural Network (CNN) trained for Malaria Detection")
    
    # Keep the Streamlit app running
    st.stop()
    
    
if __name__ == '__main__':
    main()
