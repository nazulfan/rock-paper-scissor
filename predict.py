from ultralytics import YOLO

# 1. Muat model kustom Anda
model_path = 'runs/detect/train/weights/best.pt'
model = YOLO(model_path)

# 2. Tentukan sumber video
# Untuk file video: source_video = 'path/ke/video/anda.mp4'
# Untuk webcam: source_video = 0 (angka 0 biasanya untuk webcam utama)
source_video = 0 

# 3. Lakukan prediksi dan tampilkan hasilnya secara real-time
# show=True akan membuka jendela dan menampilkan video dengan deteksi
results = model.predict(source=source_video, show=True, stream=True)

# Loop ini diperlukan agar skrip tetap berjalan saat memproses video/stream
for r in results:
    # Anda bisa mengakses hasil deteksi di sini jika perlu
    boxes = r.boxes