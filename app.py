import streamlit as st
import numpy as np
import pandas as pd
import joblib
import librosa
import os
import tempfile
from utils.predict import predict_sound
from st_audiorec import st_audiorec

# ============================
# Load model
# ============================
VOICE_MODEL_PATH = "models/voice_recognizer.pkl"
SOUND_MODEL_PATH = "models/classifier.pkl"

voice_model = joblib.load(VOICE_MODEL_PATH)
sound_model = joblib.load(SOUND_MODEL_PATH)

st.set_page_config(page_title="Voice-Activated Sound Identifier", layout="wide")

st.title("🎤 Identifikasi Suara & Autentikasi Pengguna")
st.markdown("""
Aplikasi ini hanya mengizinkan **2 orang terdaftar** untuk memberikan input suara.  
Kamu dapat **merekam langsung** dari mikrofon atau **mengunggah file suara (.wav)**.
""")

# ============================
# Pilihan Input Suara
# ============================
option = st.radio("Pilih metode input suara:", ["🎙️ Rekam langsung", "📁 Upload file (.wav)"])

# Inisialisasi variabel audio path
audio_data = None

if option == "🎙️ Rekam langsung":
    st.info("Tekan tombol di bawah untuk merekam suara kamu:")
    audio_bytes = st_audiorec()
    if audio_bytes is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(audio_bytes)
            audio_data = temp_audio.name
        st.success("✅ Rekaman berhasil! Lanjut ke proses identifikasi.")
        st.audio(audio_bytes, format='audio/wav')

elif option == "📁 Upload file (.wav)":
    uploaded_file = st.file_uploader("Upload file audio (.wav)", type=["wav"])
    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(uploaded_file.getbuffer())
            audio_data = temp_audio.name
        st.audio(uploaded_file)

# ============================
# Proses jika ada audio
# ============================
if audio_data is not None:
    # Ekstraksi fitur MFCC
    with st.spinner("🎧 Mengekstraksi fitur suara..."):
        y, sr = librosa.load(audio_data, sr=None)
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20).T, axis=0)

    # Prediksi siapa pembicara
    with st.spinner("🔍 Mengenali siapa yang berbicara..."):
        speaker_pred = voice_model.predict([mfcc])[0]

    if speaker_pred not in ["user1", "user2"]:
        st.error("🚫 Suara tidak dikenali. Hanya dua pengguna terdaftar yang diizinkan.")
    else:
        st.success(f"✅ Suara dikenali sebagai **{speaker_pred.upper()}**")

        # Prediksi jenis suara
        with st.spinner("🎯 Memprediksi jenis suara..."):
            features = predict_sound(audio_data)
            sound_pred = sound_model.predict([features])[0]

        st.subheader("📊 Hasil Prediksi:")
        st.success(f"🎯 Jenis suara: **{sound_pred.upper()}**")

        st.subheader("🔬 Fitur Statistik (dari suara):")
        st.dataframe(pd.DataFrame([features]))

    # Hapus file sementara
    os.remove(audio_data)
