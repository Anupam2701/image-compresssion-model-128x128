#!/usr/bin/env python
# coding: utf-8

# # Image Compression Model

# Import Libraries

# In[1]:


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


# Load and Preprocess Dataset

# In[2]:


import os

# List the files and directories in the unzipped dataset
dataset_path = '/kaggle/input/celeba-dataset'
print(os.listdir(dataset_path))


# In[3]:


from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import numpy as np

# Path to the directory containing images
image_dir = '/kaggle/input/celeba-dataset/img_align_celeba/img_align_celeba'

# Load images
image_size = (128, 128)
images = []
for img_name in os.listdir(image_dir)[:25000]:  # Limit to 10,000 images for memory management
    img_path = os.path.join(image_dir, img_name)
    if os.path.isfile(img_path):  # Check if the path is a file
        img = load_img(img_path, target_size=image_size)
        img = img_to_array(img) / 255.0  # Normalize to [0, 1]
        images.append(img)

images = np.array(images)


# SPLIT THE MODEL

# In[4]:


X_train, X_test = train_test_split(images, test_size=0.2, random_state=42)


# Build AutoEncoder Model

# In[5]:


def build_encoder(input_shape):
    inputs = tf.keras.Input(shape=input_shape)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    encoded = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(x)
    return models.Model(inputs, encoded, name='encoder')


# In[6]:


def build_decoder(encoded_shape):
    encoded_input = tf.keras.Input(shape=encoded_shape)
    x = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(encoded_input)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.UpSampling2D((2, 2))(x)
    decoded = layers.Conv2D(3, (3, 3), activation='sigmoid', padding='same')(x)
    
    return models.Model(encoded_input, decoded, name='decoder')


# MODEL SUMMARY

# In[7]:


input_shape = (128, 128, 3)  # For 128x128 images with 3 channels (RGB)
encoder = build_encoder(input_shape)
decoder = build_decoder(encoder.output_shape[1:])

autoencoder = models.Model(encoder.input, decoder(encoder.output))
autoencoder.compile(optimizer='adam', loss='mse')
autoencoder.summary()


# TRAIN THE MODEL

# In[8]:


history = autoencoder.fit(
    X_train, X_train,
    epochs=50,
    batch_size=64,
    shuffle=True,
    validation_data=(X_test, X_test)
)


# In[9]:


autoencoder.save('autoencoder.h5')


'''''
import tensorflow as tf

def compute_mse(original, reconstructed):
    return tf.reduce_mean(tf.square(original - reconstructed))

def compute_psnr(original, reconstructed):
    mse = compute_mse(original, reconstructed)
    psnr = 20 * tf.math.log(1.0 / tf.math.sqrt(mse)) / tf.math.log(10.0)
    return psnr


# In[13]:


reconstructed_imgs = autoencoder.predict(X_test)

mse_list = []
psnr_list = []

for i in range(len(X_test)):
    mse = compute_mse(X_test[i], reconstructed_imgs[i])
    psnr = compute_psnr(X_test[i], reconstructed_imgs[i])
    mse_list.append(mse.numpy())
    psnr_list.append(psnr.numpy())

avg_mse = np.mean(mse_list)
avg_psnr = np.mean(psnr_list)

print(f"Average MSE: {avg_mse}")
print(f"Average PSNR: {avg_psnr} dB")


# In[15]:


def compute_compression_ratio(input_shape, encoded_shape):
    input_size = np.prod(input_shape)
    encoded_size = np.prod(encoded_shape)
    compression_ratio = input_size / encoded_size
    return compression_ratio

input_shape = X_test[0].shape
encoded_shape = encoder.output_shape[1:]  # Exclude batch size

compression_ratio = compute_compression_ratio(input_shape, encoded_shape)
print(f"Compression Ratio: {compression_ratio:.2f}")

'''


