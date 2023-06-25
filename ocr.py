from image_preprocess import preprocessing_image
from image_display import display_img
from prediction import predict_ocr
import cv2
import streamlit as st
from PIL import Image
import numpy as np

# Create a file uploader
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

# If a file is uploaded
if uploaded_file is not None:
    # Open and display the image using cv2 for predicton
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    baseImg = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    # Open and display the image using PIL for displaying the image
    pil_image = Image.open(uploaded_file)
    np_image_pil = np.array(pil_image)
    # Resize the image to a fixed size
    image_view = Image.fromarray(np_image_pil).resize((300, 300))
    
    # Display the image in the sidebar
    st.sidebar.image(image_view, caption='Uploaded Image')
    prediction=predict_ocr(baseImg)
    st.write(" رقم البطاقة :", prediction)

    