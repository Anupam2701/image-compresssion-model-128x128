''''
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO

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
'''
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO

# Load your trained model
model = tf.keras.models.load_model('autoencoder.h5', compile=False)
model.compile(optimizer='adam', loss='mse')  # Recompile with the MSE loss

st.title("Image Compression using Autoencoder")

# Function to preprocess the image
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

# Function to calculate MSE
def calculate_mse(original, compressed):
    return np.mean((original - compressed) ** 2)

# Function to calculate PSNR
def calculate_psnr(original, compressed):
    mse_value = calculate_mse(original, compressed)
    if mse_value == 0:
        return float('inf')
    max_pixel = 255.0
    return 20 * np.log10(max_pixel / np.sqrt(mse_value))

# Function to calculate Compression Ratio (CR)
def calculate_cr(original_size, compressed_size):
    return original_size / compressed_size

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess the image for your model
    processed_image = preprocess_image(image)

    # Perform compression using the autoencoder model
    compressed_image = model.predict(processed_image)

    # Remove batch dimension and scale back to [0, 255] for display
    compressed_image = np.squeeze(compressed_image, axis=0)  # Remove batch dimension
    compressed_image = (compressed_image * 255).astype(np.uint8)  # Scale to [0, 255]

    # Display the compressed image
    st.image(compressed_image, caption='Compressed Image', use_column_width=True)

    # Convert the compressed image to PIL format for further processing
    compressed_image_pil = Image.fromarray(compressed_image)

    # Calculate performance parameters
    original_image_array = np.array(image.resize((128, 128)))
    mse_value = calculate_mse(original_image_array, compressed_image)
    psnr_value = calculate_psnr(original_image_array, compressed_image)

    # Calculate the original and compressed image sizes
    original_size = uploaded_file.size
    buf = BytesIO()
    compressed_image_pil.save(buf, format="JPEG")
    compressed_size = buf.tell()

    cr_value = calculate_cr(original_size, compressed_size)

    # Display performance parameters
    st.write(f"MSE: {mse_value}")
    st.write(f"PSNR: {psnr_value} dB")
    st.write(f"Compression Ratio: {cr_value}")

    # Add download button
    st.download_button(
        label="Download Compressed Image",
        data=buf.getvalue(),
        file_name="compressed_image.jpg",
        mime="image/jpeg"
    )


