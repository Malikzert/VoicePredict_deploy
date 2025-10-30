import numpy as np
import librosa
from utils.preprocess import preprocess_features

def predict_sound(file_path):
    y, sr = librosa.load(file_path, sr=None)
    
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    rmse = librosa.feature.rms(y=y)[0]
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    # Buat dict fitur mentah
    features = {
        'zcr_mean': np.mean(zcr),
        'zcr_std': np.std(zcr),
        'rmse_mean': np.mean(rmse),
        'rmse_std': np.std(rmse),
        'centroid_mean': np.mean(centroid),
        'centroid_std': np.std(centroid),
        'bandwidth_mean': np.mean(bandwidth),
        'bandwidth_std': np.std(bandwidth),
        'rolloff_mean': np.mean(rolloff),
        'rolloff_std': np.std(rolloff)
    }

    for i in range(1, 14):
        features[f'mfcc{i}_mean'] = np.mean(mfcc[i-1])
        features[f'mfcc{i}_std'] = np.std(mfcc[i-1])

    # Preprocessing sebelum prediksi
    processed_features = preprocess_features(features)
    return processed_features
