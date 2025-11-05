# ===========================================
# train_voice_model.py (RandomForest + MFCC_ + 60 Fitur)
# ===========================================
import os
import numpy as np
import pandas as pd
import librosa
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# ============================
# Konfigurasi Path
# ============================
DATASET_DIR = "data/voice_dataset"
OUTPUT_CSV = "data/voice_features.csv"
MODEL_PATH = "models/voice_recognizer.pkl"

# ============================
# Ekstraksi Fitur MFCC (60 dimensi)
# ============================
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=None)
    y, _ = librosa.effects.trim(y)
    y = librosa.util.normalize(y)

    # MFCC dasar + delta + delta2
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)

    # Ambil mean dan std untuk tiap fitur
    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)
    delta_mean = np.mean(mfcc_delta, axis=1)
    delta_std = np.std(mfcc_delta, axis=1)
    delta2_mean = np.mean(mfcc_delta2, axis=1)
    delta2_std = np.std(mfcc_delta2, axis=1)

    # Gabungkan (total 120 fitur: 6 blok × 20)
    features = np.concatenate([mfcc_mean, mfcc_std, delta_mean, delta_std, delta2_mean, delta2_std])
    return features


# ============================
# Looping Dataset
# ============================
data = []
for user in os.listdir(DATASET_DIR):
    user_folder = os.path.join(DATASET_DIR, user)
    if not os.path.isdir(user_folder):
        continue
    for file in os.listdir(user_folder):
        if file.endswith(".wav"):
            path = os.path.join(user_folder, file)
            try:
                features = extract_features(path)
                data.append([user] + list(features))
            except Exception as e:
                print(f"⚠️ Gagal ekstraksi {file}: {e}")

# ============================
# Simpan Dataset ke CSV
# ============================
columns = ["label"] + [f"mfcc_{i}" for i in range(1, 61)]
df = pd.DataFrame(data, columns=columns)
df.to_csv(OUTPUT_CSV, index=False)
print(f"✅ Dataset suara disimpan di {OUTPUT_CSV}")

# ============================
# Siapkan Data
# ============================
X = df.drop(columns=["label"])
y = df["label"]

# Normalisasi
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ============================
# Latih Model RandomForest
# ============================
model = RandomForestClassifier(
    n_estimators=300,
    random_state=42,
    class_weight="balanced",
)
model.fit(X_scaled, y)

print("✅ Model voice recognizer (RandomForestClassifier) berhasil dilatih!")

# Simpan Model + Scaler
joblib.dump({"model": model, "scaler": scaler}, MODEL_PATH)
print(f"💾 Model disimpan ke {MODEL_PATH}")

# ============================
# Evaluasi Confidence
# ============================
if hasattr(model, "predict_proba"):
    probs = model.predict_proba(X_scaled)
    preds = model.predict(X_scaled)
    confidences = [
        np.max(probs[i]) for i in range(len(X_scaled)) if preds[i] == y.iloc[i]
    ]

    if confidences:
        print("\n📊 Evaluasi Confidence di Dataset:")
        print(f" - Rata-rata confidence benar : {np.mean(confidences):.2f}")
        print(f" - Confidence minimum benar   : {np.min(confidences):.2f}")
        print(f" - Confidence maksimum benar  : {np.max(confidences):.2f}")
    else:
        print("\n⚠️ Tidak ada prediksi benar di dataset.")
