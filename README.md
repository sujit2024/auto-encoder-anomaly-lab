# Autoencoder Anomaly Lab — Flask Edition

A Flask web interface for the existing PyTorch deep-autoencoder anomaly detector. The ML architecture and reusable data/training/scoring pipeline are retained in `src/`; the Streamlit interface has been replaced with Flask routes and Jinja templates.

## Features

- Dashboard with source-data summary, configurable model training, saved-artifact status, metrics and Plotly charts.
- Single-transaction inference using the saved scaler, autoencoder and threshold.
- Batch CSV scoring with anomaly counts, ranked result preview and downloadable scored CSV.
- Dataset explorer with row preview, column quality summary, label distribution and numeric-feature distribution.
- Model/scaler/metadata/metrics/plot artifacts saved in `artifacts/` by the existing pipeline.
- Uses the supplied `data/sample.csv` by default; uploading a CSV in the training form switches the active dataset for the running server process.

## Requirements

Python 3.10+ is recommended. The commands below are for Linux, Ubuntu on WSL, or macOS.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Start the Flask app

From the project root (the directory containing `app.py`):

```bash
python app.py
```

Open `http://127.0.0.1:5000` in a browser. In WSL, Windows browsers can usually open that same localhost address while the Flask process is running inside WSL.

## Use the app

1. Open **Model dashboard**.
2. Keep the bundled CSV selected, or upload another CSV in the training form.
3. Choose the row sample size, epochs, batch size, learning rate, latent dimension, patience and threshold method.
4. Select **Train / retrain model**. Training is synchronous: keep the browser tab open until the request finishes. Start with 20,000 rows for the first run.
5. Review metrics and charts on the dashboard.
6. Use **Single transaction** to score a row, or **Batch scoring** to upload a CSV with every required model feature and download the scored results.
7. Use **Dataset explorer** to inspect data quality and distributions.

## Original command-line workflow

`main.py` and the modules under `src/` remain available:

```bash
python main.py
```

## Dataset expectations

The CSV should have numeric feature columns. A label is optional; recognized label names include `fraud`, `Class`, `label`, `is_fraud`, `target`, and `anomaly`. A binary label must use 0 for normal and 1 for anomaly. When a label is detected, training uses only rows labelled 0. The batch scorer requires every feature used by the trained model and rejects rows with missing or non-numeric values in those columns.

The provided sample dataset is large. Begin with **20,000 rows** and increase the sample size only as needed.

## Artifacts

Training writes `autoencoder.pt`, `scaler.pkl`, `model_metadata.json`, `metrics.json`, `loss_curve.png`, and `error_distribution.png` to `artifacts/`. Batch-scoring exports are stored under `artifacts/batch_results/`.

## Notes and limitations

- This is a local development application. Training runs inside the Flask request and will make the page wait until the run is complete. For very long training runs or multi-user deployments, move training into a background worker and persist job state outside process memory.
- The latest uploaded dataset and in-memory result are stored in the Flask process. Restarting the app clears that transient state; saved model artifacts remain available.
- `debug=True` is intended for local development only. Disable debug mode before deployment, configure a strong `SECRET_KEY`, and deploy behind a production WSGI server if exposing the application beyond your machine.
- The anomaly score is reconstruction error, not a calibrated fraud probability. A threshold crossing indicates a record for review, not proof of fraud.
