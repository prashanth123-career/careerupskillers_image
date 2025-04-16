import streamlit as st
import cv2
import face_recognition
import pytesseract
from PIL import Image
import numpy as np
import os
import tempfile

# Set up the app
st.set_page_config(page_title="Advanced Image Analysis", layout="wide")
st.title("Image Profile Analyzer")
st.warning("Note: This local version cannot access social media profiles. It can only analyze images directly.")

# Initialize known faces (you would need to pre-load these)
known_faces = {
    # Format: "name": [encoding]
    # Example (you'd need to generate these):
    # "John Doe": [face_recognition.face_encodings(face_image)[0]]
}

# Sidebar controls
with st.sidebar:
    st.header("Settings")
    use_ocr = st.checkbox("Extract Text (OCR)", True)
    use_face_rec = st.checkbox("Face Recognition", True)
    confidence = st.slider("Confidence Threshold", 0.1, 1.0, 0.6)

def extract_text(image):
    """Extract text using OCR"""
    text = pytesseract.image_to_string(image)
    return text if text.strip() else "No text found"

def recognize_face(face_image):
    """Match against known faces"""
    face_encodings = face_recognition.face_encodings(face_image)
    if not face_encodings:
        return None
    
    for name, encoding in known_faces.items():
        matches = face_recognition.compare_faces([encoding], face_encodings[0], tolerance=0.6)
        if matches[0]:
            return name
    return None

def analyze_image(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name
    
    image = cv2.imread(tmp_path)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_image)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(pil_image, caption="Original Image", use_column_width=True)
    
    results = {}
    
    # Face detection and recognition
    if use_face_rec:
        face_locations = face_recognition.face_locations(rgb_image)
        for i, (top, right, bottom, left) in enumerate(face_locations):
            face_image = rgb_image[top:bottom, left:right]
            name = recognize_face(face_image)
            
            if name:
                results[f"Face {i+1}"] = {"name": name, "type": "known"}
            else:
                results[f"Face {i+1}"] = {"type": "unknown"}
    
    # OCR text extraction
    if use_ocr:
        text = extract_text(pil_image)
        results["extracted_text"] = text
    
    # Display results
    with col2:
        st.subheader("Analysis Results")
        
        if "extracted_text" in results:
            st.write("**Extracted Text:**")
            st.code(results["extracted_text"])
        
        for face_id, data in results.items():
            if "name" in data:
                st.success(f"✅ {face_id}: Recognized as {data['name']}")
                # Simulate finding profiles (in a real app you'd need API access)
                st.write(f"Possible profiles for {data['name']}:")
                st.markdown(f"""
                - LinkedIn: `https://linkedin.com/search?q={data['name']}`
                - Facebook: `https://facebook.com/search?q={data['name']}`
                """)
            elif face_id.startswith("Face"):
                st.warning(f"⚠️ {face_id}: Unknown person")
    
    os.unlink(tmp_path)

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    analyze_image(uploaded_file)
else:
    st.info("Please upload an image to analyze")

st.markdown("""
### Important Notes:
1. To recognize specific people, you must first add their face encodings to the `known_faces` dictionary
2. Social media links are simulated - real lookup would require API access
3. Text extraction only works if the name appears visibly in the image
""")
