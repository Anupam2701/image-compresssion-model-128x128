import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load your trained model
model = tf.keras.models.load_model('autoencoder.h5', compile=False)
model.compile(optimizer='adam', loss='mse')  # Recompile with the MSE loss


st.title("Image Compression using Autoencoder")

# Define the preprocess_image function
def preprocess_image(image, target_size=(128, 128)):
    # Resize the image to the target size
    image = image.resize(target_size)
    # Convert the image to an array
    image = np.array(image)
    # Normalize the image to [0, 1] range
    image = image / 255.0
    # Add batch dimension
    image = np.expand_dims(image, axis=0)
    return image

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image for your model
    processed_image = preprocess_image(image)

    # Perform compression
    compressed_image = model.predict(processed_image)

    # Since the model's output is normalized, we'll scale it back to [0, 255] for display
    compressed_image = np.squeeze(compressed_image, axis=0)  # Remove batch dimension
    compressed_image = (compressed_image * 255).astype(np.uint8)  # Scale to [0, 255]

    # Display the compressed image
    st.image(compressed_image, caption='Compressed Image', use_column_width=True)
