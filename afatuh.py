import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from skimage.feature import hog
from skimage import color, exposure
from skimage.transform import resize
from PIL import Image, ImageDraw, ImageFont
import streamlit_drawable_canvas as stc

# Daftar huruf Korea sesuai model
hangeul_chars = ["Yu", "ae", "b", "bb", "ch", "d", "e", "eo", "eu", "g", "gg", "h", "i", "j", "k",
                 "m", "n", "ng", "o", "p", "r", "s", "ss", "t", "u", "ya", "yae", "ye", "yo"]

# Load model dengan caching
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("best_cnn_hog_model9010bismillahacc1.h5", compile=False)

model = load_model()
num_inputs = len(model.input_shape) if isinstance(model.input_shape, list) else 1
expected_shape = model.input_shape[1:] if num_inputs == 1 else model.input_shape[0][1:]

def preprocess_image(image):
    """Preprocessing sebelum masuk ke model."""
    image = image.convert("L")  # Konversi ke grayscale
    image = np.array(image)
    
    # Pastikan background putih dan tulisan hitam
    if np.mean(image) > 127:
        image = cv2.bitwise_not(image)
    
    # Thresholding sederhana
    _, binary_image = cv2.threshold(image, 128, 255, cv2.THRESH_BINARY_INV)
    
    # Resize sesuai model
    final_image = cv2.resize(binary_image, expected_shape[:2], interpolation=cv2.INTER_AREA) / 255.0
    
    # Pastikan channel sesuai dengan model
    if len(expected_shape) == 3 and expected_shape[-1] == 3:
        final_image = np.stack((final_image,) * 3, axis=-1)
    
    return np.expand_dims(final_image.astype(np.float32), axis=0), binary_image

def extract_hog_features(image):
    """Ekstraksi fitur HOG yang sesuai dengan model."""
    gray_image = color.rgb2gray(image) if image.ndim == 3 else image
    gray_image_resized = resize(gray_image, (64, 64), anti_aliasing=True, preserve_range=True)
    
    # Ekstraksi HOG
    hog_features, hog_image = hog(gray_image_resized, orientations=9, pixels_per_cell=(8, 8), 
                                  cells_per_block=(2, 2), block_norm='L2-Hys', visualize=True)
    
    # Normalisasi fitur
    hog_features /= (np.linalg.norm(hog_features) + 1e-6)
    hog_image = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    
    return np.expand_dims(hog_features.astype(np.float32), axis=0), hog_image

def generate_hangeul_image(text):
    """Menghasilkan gambar dari huruf Hangeul yang dikenali."""
    img = Image.new("RGB", (100, 100), "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("malgun.ttf", 50)  # Font Windows untuk Hangeul
    except:
        font = ImageFont.load_default()
    draw.text((10, 25), text, fill="black", font=font)
    return img

def main():
    st.title("📝 Pengenalan Tulisan Hangeul ")
    st.write("Ayo Belajar Hangeul tuliskan dicanvas!! by: Muhammad Fikri Riyanto")
    
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
            processed_image, binary_image = preprocess_image(image)
            
            st.write("Processed image shape:", processed_image.shape)
            
            if num_inputs == 2:
                hog_features, hog_visual = extract_hog_features(binary_image)
                st.write("HOG features shape:", hog_features.shape)
                
                # Pastikan input model memiliki bentuk yang sesuai
                if processed_image.shape[1:] != expected_shape or hog_features.shape[1:] != (144,):
                    st.error("Dimensi input tidak sesuai dengan model.")
                    return
                
                try:
                    prediction = model.predict([processed_image, hog_features])
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat prediksi: {e}")
                    return
            else:
                if processed_image.shape[1:] != expected_shape:
                    st.error("Dimensi input tidak sesuai dengan model.")
                    return
                try:
                    prediction = model.predict(processed_image)
                except Exception as e:
                    st.error(f"Terjadi kesalahan saat prediksi: {e}")
                    return
            
            # Ambil 3 prediksi teratas
            top_3_indices = np.argsort(prediction[0])[-3:][::-1]
            top_3_values = prediction[0][top_3_indices] * 100
            
            st.write("🔍 **Top 3 Prediksi Model:**")
            for i in range(3):
                predicted_hangeul = hangeul_chars[top_3_indices[i]]
                confidence = top_3_values[i]
                st.write(f"{i+1}. {predicted_hangeul} ({confidence:.2f}%)")
            
            predicted_hangeul = hangeul_chars[top_3_indices[0]]
            hangeul_image = generate_hangeul_image(predicted_hangeul)
            st.image(hangeul_image, caption=f"🖌 Huruf Hangeul: {predicted_hangeul}", use_container_width=False)
            
            st.image(processed_image[0], caption="📊 Gambar Input ke Model", use_container_width=True, clamp=True, channels="GRAY")
            st.image(binary_image, caption="📊 Gambar Setelah Thresholding", use_container_width=True, clamp=True, channels="GRAY")
            if num_inputs == 2:
                st.image(hog_visual, caption="📊 Ekstraksi HOG", use_container_width=True, clamp=True, channels="GRAY")

if __name__ == "__main__":
    main()
