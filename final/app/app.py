import streamlit as st
import numpy as np
import cv2
from PIL import Image
import tensorflow as tf
import os

from tensorflow.keras.layers import Layer
_original_layer_init = Layer.__init__

def _patched_layer_init(self, *args, **kwargs):
    kwargs.pop('quantization_config', None)
    _original_layer_init(self, *args, **kwargs)

Layer.__init__ = _patched_layer_init

st.set_page_config(page_title="AI Clinic - Graduation Project", layout="wide")

st.title("AI-Clinic: Chest X-Ray Segmentation & Diagnosis")
st.markdown("Upload a Chest X-Ray image. The system will first perform Lung Segmentation, followed by Disease Classification.")

@st.cache_resource
def load_ai_models():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(current_dir, "saved models")
    
    unet_path = os.path.join(models_dir, "unet_lung_model.keras")
    densenet_path = os.path.join(models_dir, "medical_classifier_densenet.keras")
    
    seg_model = tf.keras.models.load_model(unet_path, compile=False)
    cls_model = tf.keras.models.load_model(densenet_path, compile=False)
    
    return seg_model, cls_model

with st.spinner("Loading diagnostic models..."):
    unet_model, classifier_model = load_ai_models()

CATEGORIES = ['COVID-19', 'Lung Opacity', 'Normal (Healthy)', 'Viral Pneumonia']

uploaded_file = st.file_uploader("Upload Chest X-Ray (JPG/PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    original_image = Image.open(uploaded_file)
    
    # 1. Image Preprocessing (Grayscale for U-Net)
    img_gray = original_image.convert('L')
    img_array = np.array(img_gray)
    
    # 2. Segmentation Phase
    img_seg = cv2.resize(img_array, (256, 256))
    img_seg_norm = np.expand_dims(img_seg, axis=(0, -1)) / 255.0
    
    pred_mask = unet_model.predict(img_seg_norm)[0]
    binary_mask = (pred_mask > 0.5).astype(np.uint8)
    
    # 3. Apply Mask for Classification (Matching Kaggle exact logic)
    img_for_cls = cv2.resize(img_array, (224, 224))
    mask_for_cls = cv2.resize(binary_mask, (224, 224))
    
    segmented_lung = img_for_cls * mask_for_cls
    segmented_lung_3c = cv2.cvtColor(segmented_lung, cv2.COLOR_GRAY2RGB) / 255.0
    cls_input = np.expand_dims(segmented_lung_3c, axis=0)
    
    # 4. Final Classification (DenseNet121)
    predictions = classifier_model.predict(cls_input)[0]
    predicted_class_index = np.argmax(predictions) 
    confidence = np.max(predictions) * 100 
    diagnosis = CATEGORIES[predicted_class_index] 

    # 5. Display Results
    st.markdown("---")
    st.header(f"Final Diagnosis: {diagnosis} (Confidence: {confidence:.2f}%)")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image(original_image, caption="1. Original X-Ray", use_container_width=True)
    with col2:
        st.image(binary_mask * 255, caption="2. Lung Mask (Segmentation)", use_container_width=True)
    with col3:
        st.image(segmented_lung_3c, caption="3. Isolated Lungs (Input to DenseNet)", use_container_width=True)