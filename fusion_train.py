import pickle

import librosa
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


def extract_audio_features(file_path):
    audio, sample_rate = librosa.load(file_path, sr=22050)
    mfcc = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    return np.mean(mfcc.T, axis=0)


emotion_texts = {
    "angry": "I am angry and upset",
    "disgust": "I feel disgusted and uncomfortable",
    "fear": "I am scared and afraid",
    "happy": "I am happy and joyful",
    "neutral": "I am feeling normal",
    "pleasant_surprise": "I am surprised in a pleasant way",
    "sad": "I am sad and unhappy"
}

df = pd.read_csv("dataset.csv")
df["text"] = df["emotion"].map(emotion_texts)

audio_features = []
labels = []

print("Extracting audio features...")

for _, row in df.iterrows():
    try:
        features = extract_audio_features(row["file_path"])
        audio_features.append(features)
        labels.append(row["emotion"])
    except Exception as error:
        print("Skipped:", row["file_path"])
        print(error)

audio_features = np.array(audio_features)
text_data = df["text"][:len(audio_features)]

vectorizer = TfidfVectorizer()
text_features = vectorizer.fit_transform(text_data)

fusion_features = hstack([audio_features, text_features])

x_train, x_test, y_train, y_test = train_test_split(
    fusion_features,
    labels,
    test_size=0.2,
    random_state=42,
    stratify=labels
)

model = RandomForestClassifier(n_estimators=100, random_state=42)

print("Training fusion model...")
model.fit(x_train, y_train)

predictions = model.predict(x_test)

accuracy = accuracy_score(y_test, predictions)

print("\nFusion Model Accuracy:", accuracy)
print("\nClassification Report:\n")
print(classification_report(y_test, predictions))

with open("fusion_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

with open("fusion_vectorizer.pkl", "wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("\nFusion model saved successfully.")