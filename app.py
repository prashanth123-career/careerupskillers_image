import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

st.title("Basic Image Analysis")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name
    
    image = cv2.imread(tmp_path)
    
    if image is None:
        st.error("Could not read the image")
    else:
        st.image(image, caption="Uploaded Image", use_column_width=True)
        
        # Basic image analysis
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml').detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        
        if len(faces) > 0:
            st.success(f"Found {len(faces)} face(s) in the image")
        else:
            st.warning("No faces detected")
    
    os.unlink(tmp_path)
