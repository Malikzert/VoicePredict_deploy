import pandas as pd
import numpy as np
import joblib
import os

# Path scaler (jika digunakan)
SCALER_PATH = "models/scaler.pkl"

def preprocess_features(feature_dict):
    """
    Melakukan preprocessing fitur sebelum prediksi.
    - Pastikan urutan kolom sama dengan saat training.
    - Terapkan scaler jika ada (opsional).
    """
    # Ubah dict menjadi DataFrame
    df = pd.DataFrame([feature_dict])

    # Urutan kolom fitur sesuai training
    expected_cols = [
        'zcr_mean', 'zcr_std', 'rmse_mean', 'rmse_std',
        'centroid_mean', 'centroid_std', 'bandwidth_mean', 'bandwidth_std',
        'rolloff_mean', 'rolloff_std'
    ]

    # Tambahkan MFCC
    for i in range(1, 14):
        expected_cols.append(f'mfcc{i}_mean')
        expected_cols.append(f'mfcc{i}_std')

    # Pastikan semua kolom ada
    for col in expected_cols:
        if col not in df.columns:
            df[col] = 0.0

    # Urutkan kolom sesuai urutan yang diharapkan
    df = df[expected_cols]

    # Terapkan scaler jika tersedia
    if os.path.exists(SCALER_PATH):
        scaler = joblib.load(SCALER_PATH)
        df_scaled = scaler.transform(df)
        return df_scaled[0]
    else:
        return df.values[0]
