import pickle

import librosa
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, sr=22050)

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=sample_rate,
        n_mfcc=40
    )

    mfcc_mean = np.mean(mfcc.T, axis=0)

    return mfcc_mean


def main():

    df = pd.read_csv("dataset.csv")

    features_list = []
    labels_list = []

    print("Extracting features from audio files...")

    for _, row in df.iterrows():

        try:
            feature = extract_features(row["file_path"])

            features_list.append(feature)
            labels_list.append(row["emotion"])

        except Exception as error:

            print("Skipped file:", row["file_path"])
            print(error)

    x = np.array(features_list)
    y = np.array(labels_list)

    print("\nFeature Shape:", x.shape)

    encoder = LabelEncoder()

    y_encoded = encoder.fit_transform(y)

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y_encoded,
        test_size=0.2,
        random_state=42,
        stratify=y_encoded
    )

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    print("\nTraining Speech Emotion Model...")

    model.fit(x_train, y_train)

    predictions = model.predict(x_test)

    accuracy = accuracy_score(y_test, predictions)

    print("\nSpeech Model Accuracy:", accuracy)

    print("\nClassification Report:\n")

    print(
        classification_report(
            y_test,
            predictions,
            target_names=encoder.classes_
        )
    )

    with open("speech_model.pkl", "wb") as model_file:
        pickle.dump(model, model_file)

    with open("speech_label_encoder.pkl", "wb") as encoder_file:
        pickle.dump(encoder, encoder_file)

    print("\nModel saved successfully.")


if __name__ == "__main__":
    main()