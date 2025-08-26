from flask import Flask, request, jsonify, render_template
import torch
import psycopg2
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from ultralytics import YOLO

# Inisialisasi Flask
app = Flask(__name__)

# Load model YOLO
model = YOLO("best.pt")

# Konfigurasi koneksi PostgreSQL
def get_db_connection():
    conn = psycopg2.connect(
        dbname="pengendali_penyakit_jagung",
        user="postgres",        # ganti kalau pakai user lain
        password="mamahkuBaik4",  # ganti dengan password PostgreSQL kamu
        host="localhost",
        port="5432"            # ganti kalau pakai port lain
    )
    return conn

@app.route('/')
def home():
    return render_template('home.html')

@app.route("/about")
def about(): 
    return render_template("about.html")

@app.route("/deteksi")
def deteksi():
    return render_template("scan.html")

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
    result = results[0]

    # Ambil hasil probabilitas
    probs = result.probs  
    class_id = int(probs.top1)
    confidence = float(probs.top1conf)
    label = result.names[class_id]  # hasil penyakit

    # === Query ke PostgreSQL berdasarkan label penyakit ===
    conn = get_db_connection()
    cur = conn.cursor()

    query = """
        SELECT agen_hayati, konsentrasi, metode_aplikasi, efektivitas
        FROM pengendali_penyakit_jagung
        WHERE penyakit ILIKE %s;
    """
    cur.execute(query, (label,))
    rows = cur.fetchall()

    cur.close()
    conn.close()

    rekomendasi = []
    for r in rows:
        rekomendasi.append({
            "agen_hayati": r[0],
            "konsentrasi": r[1],
            "metode_aplikasi": r[2],
            "efektivitas": r[3]
        })

    return jsonify({
        "prediction": label,
        "confidence": confidence,
        "rekomendasi": rekomendasi
    })

if __name__ == "__main__":
    app.run(debug=True)

