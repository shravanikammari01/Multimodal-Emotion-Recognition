import os
import pandas as pd

DATASET_PATH = r"dataset"

def get_emotion_from_filename(filename):
    filename = filename.lower()

    emotions = {
        "angry": "angry",
        "disgust": "disgust",
        "fear": "fear",
        "happy": "happy",
        "neutral": "neutral",
        "ps": "pleasant_surprise",
        "sad": "sad"
    }

    for key, emotion in emotions.items():
        if key in filename:
            return emotion

    return "unknown"


def load_dataset(dataset_path):
    data = []

    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                emotion = get_emotion_from_filename(file)
                data.append([file_path, emotion])

    df = pd.DataFrame(data, columns=["file_path", "emotion"])
    return df


if __name__ == "__main__":
    df = load_dataset(DATASET_PATH)

    print("Total audio files:", len(df))
    print("\nEmotion counts:")
    print(df["emotion"].value_counts())

    df.to_csv("dataset.csv", index=False)
    print("\nSaved dataset.csv successfully.")