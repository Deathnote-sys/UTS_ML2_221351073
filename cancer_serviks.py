import streamlit as st
import tensorflow as tf
import numpy as np
import joblib
import pandas as pd

# Load scaler and model
scaler = joblib.load('scaler.pkl')
interpreter = tf.lite.Interpreter(model_path="cancer_serviks_model.tflite")
interpreter.allocate_tensors()

# Get model input/output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Define the exact feature order expected by the scaler and model
FEATURE_ORDER = [
    'Age',
    'Number of sexual partners',
    'Num of pregnancies',
    'Smokes',
    'Hormonal Contraceptives',
    'STDs',
    'STDs:HPV',
    'Citology'
]

# App title
st.title("Sistem Prediksi Risiko Kanker Serviks")
st.write("Masukkan informasi pasien untuk menilai risiko kanker serviks.")

# User input form
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Usia", min_value=15, max_value=100, value=30)
    num_partners = st.number_input("Jumlah Pasangan Seksual", min_value=0, max_value=50, value=1)
    pregnancies = st.number_input("Jumlah Kehamilan", min_value=0, max_value=20, value=0)
    smokes = st.selectbox("Merokok?", ["Tidak", "Ya"])
    
with col2:
    hormones = st.selectbox("Menggunakan Kontrasepsi Hormonal?", ["Tidak", "Ya"])
    stds = st.selectbox("Riwayat Penyakit Menular Seksual?", ["Tidak", "Ya"])
    hpv = st.selectbox("Positif HPV?", ["Tidak", "Ya"])
    citology = st.selectbox("Hasil Sitologi", ["Normal", "Abnormal"])
    hinselmann = st.selectbox("Hasil Pemeriksaan Hinselmann", ["Normal", "Abnormal"])


# Convert categorical inputs to numerical
input_mapping = {
    "Tidak": 0,
    "Ya": 1,
    "Normal": 0,
    "Abnormal": 1
}

if st.button("Prediksi Risiko"):
    try:
        # Prepare input data in the EXACT order expected by the model
        input_values = [
            age,                                   # Age
            num_partners,                          # Number of sexual partners
            pregnancies,                           # Num of pregnancies
            input_mapping[smokes],                 # Smokes
            input_mapping[hormones],               # Hormonal Contraceptives
            input_mapping[stds],                   # STDs
            input_mapping[hpv],                    # STDs:HPV
            input_mapping[citology],                # Citology
            input_mapping[hinselmann]
        ]
        
        # Convert to numpy array and reshape for scaling
        input_array = np.array(input_values, dtype=np.float32).reshape(1, -1)
        
        # Scale features
        input_scaled = scaler.transform(input_array)
        
        # Make prediction
        interpreter.set_tensor(input_details[0]['index'], input_scaled)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])
        
        probability = prediction[0][0]
        risk_level = "TINGGI" if probability > 0.5 else "RENDAH"
        
        # Display results
        st.subheader("Hasil Prediksi")
        
        if risk_level == "TINGGI":
            st.error(f"Risiko Kanker Serviks: {risk_level} ({probability*100:.1f}%)")
            st.warning("Rekomendasi: Segera konsultasikan dengan dokter spesialis kandungan untuk pemeriksaan lebih lanjut.")
        else:
            st.success(f"Risiko Kanker Serviks: {risk_level} ({probability*100:.1f}%)")
            st.info("Rekomendasi: Tetap lakukan pemeriksaan rutin sesuai anjuran dokter.")
        
        # Show probability meter
        st.progress(float(probability))
        st.write(f"Skor Risiko: {probability*100:.1f}%")
        
    except Exception as e:
        st.error(f"Terjadi kesalahan: {str(e)}")
        st.write("Pastikan semua input telah diisi dengan benar.")