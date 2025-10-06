import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os


import tensorflow.keras.backend as K
from tensorflow.keras.metrics import BinaryAccuracy, MeanIoU

# Define custom objects needed for loading the model
# This should match the custom objects used during evaluation
def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    dice_coeff = (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    return 1 - dice_coeff

def combined_loss(y_true, y_pred, alpha=0.5):
    bce = tf.keras.losses.BinaryCrossentropy()(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return alpha * bce + (1 - alpha) * dice

def dice_coef_metric(y_true, y_pred, smooth=1e-6):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


custom_objects = {
    'combined_loss': combined_loss,
    'dice_loss': dice_loss,
    'dice_coef_metric': dice_coef_metric,
    'BinaryAccuracy': BinaryAccuracy,
    'MeanIoU': MeanIoU
}

# Define the path to the saved model
# Make sure this path is accessible in your deployment environment
best_model_path = "Khushi/models/best_oil_spill_model.keras"

# Define target dimensions (should match the dimensions used for training)
IMG_WIDTH = 256
IMG_HEIGHT = 256

# Load the trained model (use st.cache_resource for efficiency)

def load_model(best_model_path, _custom_obj):
    from tensorflow.keras.models import load_model as keras_load_model
    model = keras_load_model(best_model_path, custom_objects=_custom_obj)
    return model

oil_spill_model = load_model(best_model_path, custom_objects)


st.title("Oil Spill Detection App")

uploaded_file = st.file_uploader("Upload a satellite image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Detect Oil Spill"):
        if oil_spill_model is not None:
            try:
                # Preprocess the image for prediction
                img_gray = image.convert('L') # Convert to grayscale
                img_gray = img_gray.resize((IMG_WIDTH, IMG_HEIGHT))
                img_array = np.array(img_gray)
                img_input = np.expand_dims(img_array / 255.0, axis=0)  # Normalize and add batch dimension
                img_input = np.expand_dims(img_input, axis=-1)  # Add channel dimension for grayscale

                # Perform inference
                predictions = oil_spill_model.predict(img_input)

                # Postprocess prediction
                predicted_mask = (predictions > 0.5).astype(np.uint8)
                predicted_mask = np.squeeze(predicted_mask)  # Remove batch and channel dimensions

                # Display the predicted mask
                mask_img = Image.fromarray(predicted_mask * 255, mode='L') # Scale to 0-255 for grayscale image
                st.image(mask_img, caption="Predicted Oil Spill Mask", use_container_width=True)

                # Optional: Overlay the mask on the original image for better visualization
                original_img_rgb = image.convert('RGB').resize((IMG_WIDTH, IMG_HEIGHT))
                overlay = np.zeros_like(np.array(original_img_rgb), dtype=np.uint8)
                overlay[predicted_mask == 1] = [255, 0, 0] # Red color for predicted oil spill
                blended_image = Image.blend(original_img_rgb, Image.fromarray(overlay, mode='RGB'), alpha=0.5)
                st.image(blended_image, caption="Predicted Mask Overlay", use_container_width=True)


            except Exception as e:
                st.error(f"Prediction failed: {e}")
        else:
            st.warning("Model not loaded. Cannot perform prediction.")
