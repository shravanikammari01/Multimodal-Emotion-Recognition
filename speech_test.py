import pickle

import librosa
import numpy as np


def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, sr=22050)

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=40
    )

    return np.mean(mfcc.T, axis=0)


with open("speech_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("speech_label_encoder.pkl", "rb") as encoder_file:
    encoder = pickle.load(encoder_file)


audio_path = input("Enter audio file path: ")

features = extract_features(audio_path)
features = features.reshape(1, -1)

prediction = model.predict(features)
emotion = encoder.inverse_transform(prediction)

print("\nPredicted Emotion:", emotion[0])