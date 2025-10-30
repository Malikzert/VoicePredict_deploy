import streamlit as st
import pandas as pd
import numpy as np
import joblib
import librosa
import os
from utils.predict import predict_sound

st.set_page_config(page_title="Identifikasi Suara Buka/Tutup", layout="wide")

st.title("🔊 Identifikasi Suara Buka/Tutup")
st.markdown("Gunakan model berbasis **fitur statistik time series** untuk mendeteksi apakah suara termasuk kategori **buka** atau **tutup**.")

# Load model
MODEL_PATH = "models/classifier.pkl"
model = joblib.load(MODEL_PATH)

# Upload file audio
uploaded_file = st.file_uploader("🎵 Upload file audio (.wav)", type=["wav"])

if uploaded_file:
    st.audio(uploaded_file)

    # Ekstraksi fitur
    with st.spinner("Ekstraksi fitur sedang dilakukan..."):
        temp_path = f"temp.wav"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        features = predict_sound(temp_path)
        os.remove(temp_path)

    # Prediksi
    pred = model.predict([features])[0]
    st.success(f"🎯 Prediksi: **{pred.upper()}**")

    # Tampilkan fitur
    st.subheader("📊 Fitur Statistik:")
    st.dataframe(pd.DataFrame([features]))
