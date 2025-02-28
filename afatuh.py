import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from skimage.feature import hog
from skimage.color import rgb2gray
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Daftar huruf Korea sesuai model
hangeul_chars = ["Yu", "ae", "b", "bb", "ch", "d", "e", "eo", "eu", "g", "gg", "h", "i", "j", "k",
                 "m", "n", "ng", "o", "p", "r", "s", "ss", "t", "u", "ya", "yae", "ye", "yo"]

# Load model dengan caching
@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model("best_cnn_hog_model9010bismillahacc1.h5", compile=False)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Fungsi untuk memproses gambar (CNN input)
def preprocess_image(image_path):
    try:
        img = load_img(image_path, target_size=(100, 100), color_mode="grayscale")
        img_tensor = img_to_array(img)
        img_tensor = np.expand_dims(img_tensor, axis=0)
        img_tensor /= 255.0  # Normalisasi
        return img_tensor
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

# Fungsi untuk mengekstrak fitur HOG (SVM input)
def extract_hog_features(image_path):
    try:
        image = cv2.imread(image_path)
        gray = rgb2gray(image)
        features, _ = hog(gray, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
        return np.array(features).reshape(1, -1)
    except Exception as e:
        st.error(f"Error extracting HOG features: {e}")
        return None

# Streamlit UI
st.title("Korean Character Detection")
st.write("Upload an image to predict the character.")

uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image_path = "image_user_converted.png"
    with open(image_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    st.write("Classifying...")
    
    img_tensor = preprocess_image(image_path)
    hog_features = extract_hog_features(image_path)
    model = load_model()
    
    if img_tensor is not None and hog_features is not None and model is not None:
        try:
            pred = model.predict([img_tensor, hog_features])  # Gunakan kedua input
            result = hangeul_chars[np.argmax(pred)]
            st.write(f"Prediction: **{result}**")
        except Exception as e:
            st.error(f"Error making prediction: {e}")
