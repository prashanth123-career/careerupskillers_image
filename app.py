import streamlit as st
import face_recognition
import pytesseract
from PIL import Image
import numpy as np
import os
import tempfile

# Set up the app
st.set_page_config(page_title="Person Identifier", layout="wide")
st.title("🔍 Person Identification Tool")
st.markdown("""
- **Step 1:** Upload an image
- **Step 2:** The system checks against known faces
- **Step 3:** Use Google Reverse Image Search (manual) for more details
""")

# ====== KNOWN DATASET (Pre-load your 10 people) ======
known_faces = {}  # Format: {"Name": face_encoding}

def load_known_faces():
    """Load your 10 people here"""
    # Example (replace with your dataset):
    known_person_img = face_recognition.load_image_file("person1.jpg")
    known_faces["John Doe"] = face_recognition.face_encodings(known_person_img)[0]

load_known_faces()  # Initialize dataset

# ====== GOOGLE REVERSE IMAGE SEARCH (Manual) ======
def google_reverse_search(image_path):
    """Helper function to guide manual Google search"""
    st.warning("No API used. Manually upload to Google Images:")
    st.markdown(f"""
    1. Go to [Google Images](https://images.google.com)
    2. Click the camera icon 📷
    3. Upload the image
    4. Check results for names/social profiles
    """)
    st.image(image_path, caption="Upload this image to Google", width=300)

# ====== OCR (Extract Text) ======
def extract_text(image):
    text = pytesseract.image_to_string(image)
    return text if text.strip() else None

# ====== FACE MATCHING ======
def recognize_face(uploaded_image):
    img = face_recognition.load_image_file(uploaded_image)
    face_locations = face_recognition.face_locations(img)
    
    if not face_locations:
        return None
    
    uploaded_encoding = face_recognition.face_encodings(img, face_locations)[0]
    
    for name, known_encoding in known_faces.items():
        match = face_recognition.compare_faces([known_encoding], uploaded_encoding, tolerance=0.5)
        if match[0]:
            return name
    return None

# ====== MAIN ANALYSIS ======
def analyze_image(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name
    
    # Check if face matches known dataset
    matched_name = recognize_face(tmp_path)
    
    if matched_name:
        st.success(f"✅ **Match Found:** {matched_name}")
        st.markdown(f"""
        **Possible Profiles:**
        - LinkedIn: `https://www.linkedin.com/search/results/all/?keywords={matched_name}`
        - Facebook: `https://www.facebook.com/search/top?q={matched_name}`
        """)
    else:
        st.warning("❌ No match in local database")
        st.info("Try Google Reverse Image Search for more info:")
        google_reverse_search(tmp_path)
    
    # Extract text (OCR)
    extracted_text = extract_text(Image.open(tmp_path))
    if extracted_text:
        st.markdown("**Extracted Text (OCR):**")
        st.code(extracted_text)
    
    os.unlink(tmp_path)

# ====== FILE UPLOAD ======
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    analyze_image(uploaded_file)
else:
    st.info("Please upload an image to analyze")

st.markdown("---")
st.subheader("How to Use:")
st.markdown("""
1. **For your 10 known people**: Add their images in the code under `load_known_faces()`
2. **For unknown faces**: Manually use Google Reverse Image Search
3. **For text extraction**: OCR will detect visible names
""")
