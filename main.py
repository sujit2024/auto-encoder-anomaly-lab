from src.data_preprocessing import load_data, preprocess
from src.model import build_autoencoder
from src.train import train_model
from src.evaluate import compute_threshold, detect_anomalies, evaluate_predictions, reconstruction_error
from src.utils import plot_loss, plot_error_distribution
import numpy as np

def main():
    # Load and preprocess
    df = load_data()
    X_train, X_test, y_test, scaler, label_col = preprocess(df)

    # Build model
    model = build_autoencoder(X_train.shape[1])

    # Train
    history = train_model(model, X_train)

    # Plot loss
    plot_loss(history)

    # Reconstruction error
    train_err = reconstruction_error(model, X_train)
    test_err = reconstruction_error(model, X_test)

    # Threshold
    threshold = np.percentile(train_err, 99)

    # Detect anomalies
    anomalies = detect_anomalies(test_err, threshold)

    print(f"Threshold: {threshold}")
    print(f"Detected anomalies: {sum(anomalies)}")

    if y_test is not None and label_col is not None:
        metrics = evaluate_predictions(y_test, anomalies.astype(int))
        print(f"Label column: {label_col}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1 score: {metrics['f1']:.4f}")
        print("Confusion matrix:")
        print(metrics["confusion_matrix"])

    # Plot error distribution
    plot_error_distribution(train_err, test_err, threshold)

if __name__ == "__main__":
    main()