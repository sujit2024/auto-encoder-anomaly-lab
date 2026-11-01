import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from config import THRESHOLD_STD_FACTOR


def reconstruction_error(model, X):
    X = np.asarray(X, dtype=np.float32)
    X_pred = model.predict(X)
    error = np.mean((X - X_pred) ** 2, axis=1)
    return error


def compute_threshold(train_errors):
    mean = np.mean(train_errors)
    std = np.std(train_errors)
    threshold = mean + THRESHOLD_STD_FACTOR * std
    return threshold


def detect_anomalies(errors, threshold):
    return errors > threshold


def evaluate_predictions(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
    }