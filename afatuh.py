import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from skimage.feature import hog
from skimage.color import rgb2gray
from skimage.transform import resize
from PIL import Image
import streamlit_drawable_canvas as stc

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

def preprocess_image(image):
    image = image.convert("L")  # Convert to grayscale
    image = image.resize((28, 28))  # Resize to match model input size
    img_tensor = np.array(image) / 255.0  # Normalize
    img_tensor = np.expand_dims(img_tensor, axis=(0, -1))
    img_tensor = np.repeat(img_tensor, 3, axis=-1)  # Ensure 3 channels if needed
    return img_tensor

def extract_hog_features(image):
    gray = rgb2gray(image)
    gray_resized = resize(gray, (64, 64), anti_aliasing=True)
    features, _ = hog(gray_resized, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True)
    target_hog_size = 144
    features = np.pad(features, (0, max(0, target_hog_size - len(features))))[:target_hog_size]
    return np.array(features).reshape(1, -1)

st.title("📝 Pengenalan Tulisan Hangeul")
st.write("Gambar huruf di kanvas untuk prediksi.")

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

if st.button("Prediksi"):
    if canvas_result.image_data is not None:
        image = Image.fromarray((canvas_result.image_data[:, :, :3]).astype(np.uint8))
        processed_image = preprocess_image(image)
        hog_features = extract_hog_features(np.array(image))
        model = load_model()
        
        if model is not None:
            try:
                input_shapes = model.input_shape if isinstance(model.input_shape, list) else [model.input_shape]
                if len(input_shapes) == 2:
                    pred = model.predict([processed_image, hog_features])
                else:
                    pred = model.predict(processed_image)
                
                result = hangeul_chars[np.argmax(pred)]
                st.write(f"✏️ Prediksi Huruf: **{result}**")
                
                # Tampilkan hasil preprocessing
                st.image(processed_image[0], caption="📊 Gambar Setelah Preprocessing", use_column_width=True, clamp=True, channels="RGB")
            except Exception as e:
                st.error(f"Error making prediction: {e}")
