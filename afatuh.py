import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from skimage.feature import hog
from skimage.color import rgb2gray
from skimage.transform import resize
from PIL import Image
import streamlit_drawable_canvas as stc

# Konfigurasi halaman
st.set_page_config(page_title="Hangeul Detector", page_icon="📝", layout="centered")

# Daftar huruf Korea
hangeul_chars = ["Yu", "ae", "b", "bb", "ch", "d", "e", "eo", "eu", "g", "gg", "h", "i", "j", "k",
                 "m", "n", "ng", "o", "p", "r", "s", "ss", "t", "u", "ya", "yae", "ye", "yo"]

# Load model utama
@st.cache_resource
def load_model():
    try:
        return tf.keras.models.load_model("best_cnn_hog_model9010bismillahacc1.h5", compile=False)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

def preprocess_image(image):
    image = image.convert("L")
    image = np.array(image)
    if np.mean(image) > 127:
        image = cv2.bitwise_not(image)
    _, image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    image = cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)
    img_tensor = image.astype(np.float32) / 255.0
    img_tensor = np.expand_dims(img_tensor, axis=(0, -1))
    img_tensor = np.repeat(img_tensor, 3, axis=-1)
    return img_tensor

def extract_hog_features(image):
    gray = rgb2gray(image) if image.ndim == 3 else image
    gray_resized = resize(gray, (64, 64), anti_aliasing=True)
    features, _ = hog(gray_resized, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    features = features / (np.linalg.norm(features) + 1e-6)
    target_hog_size = 144
    features = np.pad(features, (0, max(0, target_hog_size - len(features))))[:target_hog_size]
    return np.array(features).reshape(1, -1)

def is_valid_drawing(image):
    non_zero_pixels = np.count_nonzero(image)
    total_pixels = image.size
    return (non_zero_pixels / total_pixels) > 0.05

# UI Header
st.title("📝AYOO BELAJAR HURUF HANGEUL!!!by:Muhammad Fikri Riyanto")
st.markdown("<h4 style='text-align: center;'>Ayo Coret Di Kanvas :)</h4>", unsafe_allow_html=True)

# Kanvas menggambar
canvas_result = stc.st_canvas(
    fill_color="rgba(255, 255, 255, 0)",
    stroke_width=10,
    stroke_color="#000000",
    background_color="#FFFFFF",
    width=256,
    height=256,
    drawing_mode="freedraw",
    key="canvas"
)

# Tombol prediksi
if st.button("🔍 Prediksi Huruf"):
    if canvas_result.image_data is not None:
        image = Image.fromarray((canvas_result.image_data[:, :, :3]).astype(np.uint8))
        if not is_valid_drawing(np.array(image.convert("L"))):
            st.error("❌ Gambar terlalu kosong atau tidak jelas. Silakan coba lagi.")
        else:
            processed_image = preprocess_image(image)
            hog_features = extract_hog_features(np.array(image))
            model = load_model()
            
            if model is not None:
                with st.spinner("🔄 Memprediksi huruf..."):
                    try:
                        input_shapes = model.input_shape if isinstance(model.input_shape, list) else [model.input_shape]
                        if len(input_shapes) == 2:
                            pred = model.predict([processed_image, hog_features])
                        else:
                            pred = model.predict(processed_image)
                        
                        confidence = np.max(pred)
                        result = hangeul_chars[np.argmax(pred)]
                        
                        if confidence < 0.7:
                            st.error("❌ Prediksi tidak cukup yakin bahwa ini huruf Hangeul. Silakan coba lagi.")
                        else:
                            st.success(f"✏️ Prediksi Huruf: **{result}**")
                        
                        st.image(processed_image[0], caption="📊 Gambar Setelah Preprocessing", use_column_width=True, clamp=True, channels="GRAY")
                    except Exception as e:
                        st.error(f"❌ Error saat memprediksi: {e}")
