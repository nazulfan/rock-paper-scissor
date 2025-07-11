# app.py

import streamlit as st
from ultralytics import YOLO
from PIL import Image
import os
import cv2
# Impor VideoProcessorBase, bukan VideoTransformerBase
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration # <-- DIUBAH

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Deteksi Batu Gunting Kertas", page_icon="✂️", layout="wide")

# --- JUDUL APLIKASI ---
st.title("Aplikasi Deteksi Batu Gunting Kertas ✊✋✌️")
st.write(
    "Pilih mode input di bawah ini: unggah gambar statis atau gunakan webcam secara real-time."
)

# --- FUNGSI UNTUK MEMUAT MODEL (DENGAN CACHE) ---
@st.cache_resource
def load_yolo_model(model_path):
    """Memuat model YOLOv8 dari path yang diberikan."""
    model = YOLO(model_path)
    return model

# --- PATH MODEL ---
model_path = 'runs/detect/train/weights/best.pt'

if not os.path.exists(model_path):
    st.error(f"File model tidak ditemukan di '{model_path}'. Pastikan Anda sudah melatih model dan path-nya sudah benar.")
    st.stop() 

try:
    model = load_yolo_model(model_path)
except Exception as e:
    st.error(f"Gagal memuat model: {e}")
    st.stop()

# --- RTC CONFIGURATION UNTUK DEPLOYMENT ---
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# --- KELAS UNTUK PROSESOR VIDEO WEBCAM ---
# Ganti nama kelas dan basisnya
class VideoProcessor(VideoProcessorBase): # <-- DIUBAH
    def __init__(self):
        self.model = model

    def recv(self, frame):
        # Konversi frame menjadi format yang bisa diproses (BGR)
        img = frame.to_ndarray(format="bgr24")

        # Lakukan deteksi objek
        results = self.model.predict(img, verbose=False)

        # Gambar kotak deteksi pada frame
        annotated_frame = results[0].plot()
        
        # Kembalikan frame yang sudah dianotasi
        return frame.from_ndarray(annotated_frame, format="bgr24")


# --- UI TABS ---
tab1, tab2 = st.tabs(["🖼️ Upload Gambar", "📹 Webcam Real-time"])

# --- TAB UNTUK UPLOAD GAMBAR ---
with tab1:
    st.header("Deteksi dari Gambar")
    uploaded_file = st.file_uploader("Pilih sebuah gambar...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        col1, col2 = st.columns(2)
        with col1:
            image = Image.open(uploaded_file)
            st.image(image, caption="Gambar yang Diunggah", use_column_width=True)

        with col2:
            with st.spinner("Sedang memproses..."):
                results = model.predict(image, verbose=False)
                annotated_image_np = results[0].plot()
                annotated_image_rgb = cv2.cvtColor(annotated_image_np, cv2.COLOR_BGR2RGB)
                st.image(annotated_image_rgb, caption="Hasil Deteksi", use_column_width=True)

# --- TAB UNTUK WEBCAM ---
with tab2:
    st.header("Deteksi dari Webcam")
    st.write("Klik 'START' untuk menyalakan webcam dan memulai deteksi secara real-time.")

    webrtc_streamer(
        key="webcam",
        # Gunakan video_processor_factory
        video_processor_factory=VideoProcessor, # <-- DIUBAH
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False}
    )
