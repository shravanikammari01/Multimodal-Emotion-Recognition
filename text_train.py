import pickle
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


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

x = df["text"]
y = df["emotion"]

x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

vectorizer = TfidfVectorizer()
x_train_vector = vectorizer.fit_transform(x_train)
x_test_vector = vectorizer.transform(x_test)

model = LogisticRegression(max_iter=1000)

print("Training text emotion model...")
model.fit(x_train_vector, y_train)

predictions = model.predict(x_test_vector)

accuracy = accuracy_score(y_test, predictions)

print("\nText Model Accuracy:", accuracy)
print("\nClassification Report:\n")
print(classification_report(y_test, predictions))

with open("text_model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

with open("text_vectorizer.pkl", "wb") as vectorizer_file:
    pickle.dump(vectorizer, vectorizer_file)

print("\nText model saved successfully.")