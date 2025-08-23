from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from ultralytics import YOLO

# Inisialisasi Flask
app = Flask(__name__, static_folder="static", template_folder="static")

# Load model
model = YOLO("best.pt")

# Transformasi gambar (sesuaikan dengan training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),   # Sesuaikan ukuran input
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Label kelas (isi sesuai dataset training)
classes = ["Sehat", "Penyakit A", "Penyakit B"]

@app.route("/")
def home():
    return app.send_static_file("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "Tidak ada file yang diupload"})
    
    file = request.files["file"]

    # Simpan sementara lalu prediksi
    filepath = "temp.jpg"
    file.save(filepath)

    # Jalankan prediksi dengan YOLO
    results = model.predict(filepath)

    # Ambil hasil pertama
    result = results[0]

    # Ambil probabilitas & label
    probs = result.probs  # tensor probabilitas
    class_id = int(probs.top1)  # index kelas dengan probabilitas tertinggi
    confidence = float(probs.top1conf)  # confidence nilai tertinggi
    label = result.names[class_id]  # mapping index ke nama kelas

    return jsonify({
        "prediction": label,
        "confidence": confidence
    })

if __name__ == "__main__":
    app.run(debug=True)
