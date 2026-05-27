import librosa
import numpy as np
import pandas as pd

def extract_features(file_path):
    try:
        audio, sample_rate = librosa.load(file_path, sr=22050)

        mfccs = librosa.feature.mfcc(
            y=audio,
            sr=sample_rate,
            n_mfcc=40
        )

        mfccs_mean = np.mean(mfccs.T, axis=0)

        return mfccs_mean

    except Exception as e:
        print("Error processing:", file_path)
        print(e)
        return None


if __name__ == "__main__":
    df = pd.read_csv("dataset.csv")

    sample_file = df["file_path"].iloc[0]

    print("Testing audio file:")
    print(sample_file)

    features = extract_features(sample_file)

    if features is not None:
        print("\nMFCC Feature Shape:", features.shape)
        print("\nMFCC Features:")
        print(features)
    else:
        print("Feature extraction failed.")
