import pickle

import librosa
import numpy as np
from scipy.sparse import hstack


def extract_audio_features(file_path):
    audio, sample_rate = librosa.load(file_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)


with open("fusion_model.pkl", "rb") as model_file:
    model = pickle.load(model_file)

with open("fusion_vectorizer.pkl", "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)


audio_path = input("Enter audio file path: ")
text_input = input("Enter text: ")

audio_features = extract_audio_features(audio_path)
audio_features = audio_features.reshape(1, -1)

text_features = vectorizer.transform([text_input])

fusion_features = hstack([audio_features, text_features])

prediction = model.predict(fusion_features)

print("\nPredicted Emotion:", prediction[0])