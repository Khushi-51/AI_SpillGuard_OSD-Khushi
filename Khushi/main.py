import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import gdown
import tensorflow.keras.backend as K
from tensorflow.keras.metrics import BinaryAccuracy, MeanIoU

# --- Custom loss and metric functions ---
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

# --- Correct custom_objects with instances ---
custom_objects = {
    'combined_loss': combined_loss,
    'dice_loss': dice_loss,
    'dice_coef_metric': dice_coef_metric,
    'BinaryAccuracy': BinaryAccuracy,
    'MeanIoU': MeanIoU
}

# --- Paths ---
file_id = "1khwzEpt6oYbb7FQATGqyJ6slXSbyqfOF"
best_model_path = "best_oil_spill_model.keras"

# Download model if not exists
if not os.path.exists(best_model_path):
    gdown.download(f"https://drive.google.com/uc?id={file_id}", best_model_path, quiet=False)

# --- Load model  ---

def load_model_cached(best_model_path, custom_objects):
    from tensorflow.keras.models import load_model as keras_load_model
    return keras_load_model(best_model_path, custom_objects=custom_objects)

oil_spill_model = load_model_cached(best_model_path, custom_objects,compile=False)

# --- Streamlit UI ---
st.title("Oil Spill Detection App")
uploaded_file = st.file_uploader("Upload a satellite image...", type=["jpg", "jpeg", "png"])

IMG_WIDTH = 256
IMG_HEIGHT = 256

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    if st.button("Detect Oil Spill"):
        if oil_spill_model:
            try:
                # Preprocess image
                img_gray = image.convert('L').resize((IMG_WIDTH, IMG_HEIGHT))
                img_array = np.array(img_gray) / 255.0
                img_input = np.expand_dims(img_array, axis=(0, -1))  # add batch & channel

                # Predict mask
                predictions = oil_spill_model.predict(img_input)
                predicted_mask = (predictions > 0.5).astype(np.uint8)
                predicted_mask = np.squeeze(predicted_mask)

                # Display mask
                mask_img = Image.fromarray(predicted_mask * 255, mode='L')
                st.image(mask_img, caption="Predicted Oil Spill Mask", use_container_width=True)

                # Overlay on original image
                original_img_rgb = image.convert('RGB').resize((IMG_WIDTH, IMG_HEIGHT))
                overlay = np.zeros_like(np.array(original_img_rgb), dtype=np.uint8)
                overlay[predicted_mask == 1] = [255, 0, 0]  # Red for oil spill
                blended_image = Image.blend(original_img_rgb, Image.fromarray(overlay, mode='RGB'), alpha=0.5)
                st.image(blended_image, caption="Predicted Mask Overlay", use_container_width=True)

            except Exception as e:
                st.error(f"Prediction failed: {e}")
        else:
            st.warning("Model not loaded.")
