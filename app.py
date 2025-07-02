import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import img_to_array
from PIL import Image, UnidentifiedImageError
import os
from datetime import datetime
import random
import base64
import streamlit as st

# Load image and convert to base64
def add_bg_image(image_path):
    with open(image_path, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded_string}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# Call the function with image path
add_bg_image("static/images/butterfly_bg.jpg")

# Load the model
model = tf.keras.models.load_model("butterfly_model.h5")

# Define butterfly classes
class_labels = {
    0: "Monarch",
    1: "Swallowtail",
    2: "Painted Lady",
    3: "Blue Morpho"
}

# Ensure uploads directory exists
if not os.path.exists("uploads"):
    os.makedirs("uploads")

# Inject CSS for custom theme
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700&display=swap');

    .stApp {
        background-color: #e3f2fd;  /* Light blue */
        font-family: 'Nunito', sans-serif;
        color: #1a237e;
    }

    h1, h2, h3, h4, h5, h6, p, span, label {
        color: #1a237e !important;
        font-family: 'Nunito', sans-serif;
        font-weight: 600;
    }

    .stButton>button {
        background-color: #3949ab;
        color: white;
        border-radius: 8px;
        padding: 10px 20px;
        font-weight: 600;
        font-family: 'Nunito', sans-serif;
        border: none;
    }

    .stButton>button:hover {
        background-color: #1a237e;
    }

    .css-12w0qpk {
        color: #00796b !important;  /* Accent for file uploader label */
    }
    </style>
""", unsafe_allow_html=True)


# Streamlit layout
st.set_page_config(page_title="Enchanted Wings 🦋", layout="centered")
st.title("🦋Enchanted Wings: Marvels of Butterfly Species")
st.markdown("Discover the beauty of butterflies. Upload a photo to identify its **species** with AI.")

uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    try:
        # Save file
        filename = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{uploaded_file.name}"
        filepath = os.path.join("uploads", filename)
        with open(filepath, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Display image
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_container_width=True)

        # Preprocess
        img_resized = img.resize((224, 224))
        img_array = img_to_array(img_resized) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # TEMP: Simulate detection
        is_butterfly = random.choice([True, False])

        if is_butterfly:
            prediction = model.predict(img_array)
            predicted_index = np.argmax(prediction)
            predicted_class = class_labels.get(predicted_index, "Unknown")
            confidence = np.max(prediction) * 100

            st.success(f"🦋 **Predicted:** {predicted_class}")
            st.info(f"📊 **Confidence:** {confidence:.2f}%")
            st.write(f"📁 **Saved as:** `{filename}`")
        else:
            st.error(f"❌ *This doesn't look like a butterfly image.*\n📁 *File:* `{filename}`")

    except UnidentifiedImageError:
        st.error("⚠ Cannot open this image. Please upload a valid file.")
    except Exception as e:
        st.error(f"⚠ Error: {str(e)}")
#Footer
st.markdown(
    """
    <hr style="margin-top: 2em; border: 1px solid #3949ab;">
    <div style='text-align: center; color: #1a237e; font-size: 16px; font-weight: 700;'>
        © 2025 Thupakula Leena Sri
    </div>
    """,
    unsafe_allow_html=True
) 
