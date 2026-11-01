import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from config import DATA_PATH, TEST_SIZE, RANDOM_STATE


LABEL_COLUMNS = ("Class", "label", "fraud")


def load_data():
    df = pd.read_csv(DATA_PATH)
    return df


def preprocess(df):
    label_col = next((column for column in LABEL_COLUMNS if column in df.columns), None)

    if label_col is not None:
        labels = df[label_col].astype(int).to_numpy()
        features = df.drop(columns=[label_col]).select_dtypes(include=[np.number])
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            features,
            labels,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            stratify=labels if len(np.unique(labels)) > 1 else None,
        )
        X_train_raw = X_train_raw[y_train == 0]
    else:
        features = df.select_dtypes(include=[np.number])
        X_train_raw, X_test_raw = train_test_split(
            features,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
        )
        y_test = None

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    return X_train, X_test, y_test, scaler, label_col