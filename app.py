from roboflow import Roboflow
from ultralytics import YOLO
import os

# 1. Unduh Dataset dari Roboflow
# Ganti dengan API key dan detail proyek Anda yang didapat dari Roboflow
rf = Roboflow(api_key="u5vRwAeQQzEPryUdpwUu")
project = rf.workspace("g-hisd7").project("rock-paper-scissor-xyglz")
dataset = project.version(5).download("yolov8")

# Lokasi file data.yaml akan ada di dalam direktori dataset yang diunduh
# Biasanya di dataset.location + '/data.yaml'
data_yaml_path = os.path.join(dataset.location, 'data.yaml')

# 2. Inisialisasi Model YOLOv8
# Kita akan menggunakan model pre-trained 'yolov8n.pt' (n artinya nano, model terkecil)
# Model akan diunduh secara otomatis saat kode pertama kali dijalankan
model = YOLO('yolov8n.pt')

# 3. Latih Model (Training)
# 'data' merujuk ke file konfigurasi dataset (data.yaml)
# 'epochs' adalah berapa kali model akan "melihat" keseluruhan dataset
# 'imgsz' adalah ukuran gambar yang akan digunakan untuk training
results = model.train(
    data=data_yaml_path,
    epochs=50,
    imgsz=640
)

# 4. (Opsional) Evaluasi Model di Test Set
# Setelah training selesai, model terbaik ('best.pt') akan disimpan di folder 'runs/detect/train/'
# Anda bisa mengevaluasi performanya pada data test
results = model.val(data=data_yaml_path, split='test')

print("Lokasi model yang telah dilatih tersimpan di folder 'runs/detect/train/'")