import streamlit as st
import numpy as np
import pandas as pd
import joblib
import librosa
import os
import tempfile
from datetime import datetime

# Coba impor st_audiorec dari 2 kemungkinan nama modul
try:
    from st_audiorec import st_audiorec
except ModuleNotFoundError:
    from streamlit_audiorec import st_audiorec

from utils.predict import predict_sound

# ============================
# Load model
# ============================
VOICE_MODEL_PATH = "models/voice_recognizer.pkl"
SOUND_MODEL_PATH = "models/classifier.pkl"
LOG_PATH = "logs/voice_log.csv"

os.makedirs("logs", exist_ok=True)  # pastikan folder log ada

try:
    voice_model = joblib.load(VOICE_MODEL_PATH)
    sound_model = joblib.load(SOUND_MODEL_PATH)
except Exception as e:
    st.error(f"❌ Gagal memuat model: {e}")
    st.stop()

st.set_page_config(page_title="Voice-Activated Sound Identifier", layout="wide")

st.title("🎤 Identifikasi Suara & Autentikasi Pengguna")
st.markdown("""
Aplikasi ini hanya mengizinkan **2 orang terdaftar (user1 & user2)** untuk memberikan input suara.  
Kamu dapat **merekam langsung** dari mikrofon atau **mengunggah file suara (.wav)**.
""")

# ============================
# Pilihan Input Suara
# ============================
option = st.radio("Pilih metode input suara:", ["🎙️ Rekam langsung", "📁 Upload file (.wav)"])
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
    with st.spinner("🎧 Mengekstraksi fitur suara..."):
        y, sr = librosa.load(audio_data, sr=None)
        mfcc = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20).T, axis=0)

    # Prediksi siapa pembicara
    with st.spinner("🔍 Mengenali siapa yang berbicara..."):
        if hasattr(voice_model, "predict_proba"):
            probs = voice_model.predict_proba([mfcc])[0]
            speaker_pred = voice_model.classes_[np.argmax(probs)]
            confidence = np.max(probs)
        else:
            speaker_pred = voice_model.predict([mfcc])[0]
            confidence = 1.0

    # Ambang batas hasil evaluasi dataset
    CONFIDENCE_THRESHOLD = 0.85

    # Default nilai prediksi jenis suara
    sound_pred = "-"
    status = "Unknown"

    if confidence < CONFIDENCE_THRESHOLD or speaker_pred not in ["user1", "user2"]:
        st.error(f"🚫 Suara **tidak dikenali** (confidence: {confidence:.2f}).\nHanya dua pengguna terdaftar yang diizinkan.")
        status = "Unrecognized"
    else:
        st.success(f"✅ Suara dikenali sebagai **{speaker_pred.upper()}** (confidence: {confidence:.2f})")

        # Prediksi jenis suara
        with st.spinner("🎯 Memprediksi jenis suara..."):
            features = predict_sound(audio_data)
            sound_pred = sound_model.predict([features])[0]

        st.subheader("📊 Hasil Prediksi:")
        st.success(f"🎯 Jenis suara: **{sound_pred.upper()}**")

        st.subheader("🔬 Fitur Statistik (dari suara):")
        st.dataframe(pd.DataFrame([features]))

        status = "Recognized"

    # ============================
    # Simpan log ke CSV
    # ============================
    log_entry = pd.DataFrame([{
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "speaker_pred": speaker_pred,
        "confidence": round(confidence, 3),
        "sound_pred": sound_pred,
        "status": status
    }])

    if os.path.exists(LOG_PATH):
        log_entry.to_csv(LOG_PATH, mode='a', header=False, index=False)
    else:
        log_entry.to_csv(LOG_PATH, index=False)

    st.info("🧾 Log hasil disimpan di `logs/voice_log.csv`")

    # Hapus file sementara
    try:
        os.remove(audio_data)
    except Exception:
        pass
