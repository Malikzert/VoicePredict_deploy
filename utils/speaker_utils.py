import numpy as np
import librosa
import joblib

# Lokasi model pengenal suara
SPEAKER_MODEL_PATH = "models/voice_recognizer.pkl"

def extract_speaker_features(file_path):
    """Ekstraksi fitur suara untuk mengenali siapa yang berbicara"""
    try:
        y, sr = librosa.load(file_path, sr=None)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        spec_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

        features = np.hstack([
            np.mean(mfcc, axis=1),
            np.std(mfcc, axis=1),
            np.mean(chroma, axis=1),
            np.std(chroma, axis=1),
            np.mean(spec_contrast, axis=1),
            np.std(spec_contrast, axis=1)
        ])
        return features
    except Exception as e:
        print(f"⚠️ Error extracting speaker features: {e}")
        return None


def predict_speaker(file_path, threshold=0.85):
    """Prediksi siapa yang berbicara dengan confidence threshold"""
    model = joblib.load(SPEAKER_MODEL_PATH)
    features = extract_speaker_features(file_path)
    if features is None:
        return None, 0.0

    # Prediksi dan hitung confidence
    probs = model.predict_proba([features])[0]
    pred_label = model.classes_[np.argmax(probs)]
    confidence = np.max(probs)

    # Gunakan threshold dari hasil evaluasi dataset
    if confidence < threshold:
        return "Unknown", confidence
    else:
        return pred_label, confidence
