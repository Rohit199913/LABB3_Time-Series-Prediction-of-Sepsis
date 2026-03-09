import numpy as np
import pandas as pd


def create_windows(df, window_size=12, horizon_steps=4, stride=1):
    df = df.sort_values(["id", "timestep"]).copy()

    feature_columns = [col for col in df.columns if col not in ["id", "sepsis"]]

    X = []
    y = []

    for patient_id, patient_df in df.groupby("id"):
        patient_df = patient_df.sort_values("timestep").reset_index(drop=True)

        features = patient_df[feature_columns].values
        labels = patient_df["sepsis"].values

        max_start = len(patient_df) - window_size - horizon_steps + 1

        for start in range(0, max_start, stride):
            end = start + window_size
            target_index = end + horizon_steps - 1

            X_window = features[start:end]
            y_target = labels[target_index]

            X.append(X_window)
            y.append(y_target)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    return X, y, feature_columns