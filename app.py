from __future__ import annotations

import io
import json
import os
import sys
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from flask import (
    Flask,
    abort,
    flash,
    redirect,
    render_template,
    request,
    send_from_directory,
    url_for,
)
from werkzeug.utils import secure_filename

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dashboard_pipeline import (  # noqa: E402
    ARTIFACT_DIR,
    detect_label_column,
    load_saved_bundle,
    prepare_dataset,
    score_dataframe,
    train_pipeline,
)

app = Flask(__name__)
app.config.update(
    SECRET_KEY=os.environ.get("SECRET_KEY", "local-development-key"),
    MAX_CONTENT_LENGTH=300 * 1024 * 1024,  # Allow the included large CSV and reasonable uploads.
)

DATA_PATH = PROJECT_ROOT / "data" / "sample.csv"
BATCH_RESULTS_DIR = ARTIFACT_DIR / "batch_results"
ALLOWED_ARTIFACT_IMAGES = {"loss_curve.png", "error_distribution.png"}

# This is a local development application. State is kept in this process so that
# pages can share the latest uploaded dataset and training result between routes.
CURRENT_SOURCE_DF: pd.DataFrame | None = None
CURRENT_SOURCE_NAME = "data/sample.csv"
CURRENT_BUNDLE: dict[str, Any] | None = None
LAST_RESULT: dict[str, Any] | None = None
LAST_TRAINING_SUMMARY: dict[str, Any] | None = None


def load_default_dataset() -> pd.DataFrame:
    """Load the bundled dataset on first use and cache it for this process."""
    global CURRENT_SOURCE_DF
    if CURRENT_SOURCE_DF is None:
        if not DATA_PATH.exists():
            raise FileNotFoundError(f"Default dataset was not found at {DATA_PATH}.")
        CURRENT_SOURCE_DF = pd.read_csv(DATA_PATH)
    return CURRENT_SOURCE_DF


def get_bundle() -> dict[str, Any] | None:
    """Return the current in-memory model or try loading saved artifacts."""
    global CURRENT_BUNDLE
    if CURRENT_BUNDLE is None:
        CURRENT_BUNDLE = load_saved_bundle()
    return CURRENT_BUNDLE


def load_saved_metrics() -> dict[str, Any] | None:
    metrics_path = ARTIFACT_DIR / "metrics.json"
    if not metrics_path.exists():
        return None
    try:
        with metrics_path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return None


def fmt_int(value: int | float | None) -> str:
    if value is None:
        return "—"
    return f"{int(value):,}"


def normalised_label(column: str) -> str:
    return str(column).replace("_", " ").strip().title()


def build_loss_chart(history: dict[str, list[float]] | None) -> str | None:
    if not history or not history.get("train_loss"):
        return None
    epochs = list(range(1, len(history["train_loss"]) + 1))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=epochs, y=history["train_loss"], mode="lines+markers", name="Training loss"))
    if history.get("val_loss"):
        fig.add_trace(go.Scatter(x=epochs, y=history["val_loss"], mode="lines+markers", name="Validation loss"))
    fig.update_layout(
        title="Training history",
        xaxis_title="Epoch",
        yaxis_title="Mean squared error",
        height=330,
        margin=dict(l=18, r=18, t=55, b=35),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False, "responsive": True})


def build_error_chart(result: dict[str, Any] | None) -> str | None:
    if not result or "train_errors" not in result or "test_errors" not in result:
        return None
    train_errors = np.asarray(result["train_errors"], dtype=float)
    test_errors = np.asarray(result["test_errors"], dtype=float)
    # Limit chart points/bins for large datasets while leaving model calculations unchanged.
    if len(train_errors) > 20000:
        rng = np.random.default_rng(42)
        train_errors = rng.choice(train_errors, 20000, replace=False)
    if len(test_errors) > 20000:
        rng = np.random.default_rng(43)
        test_errors = rng.choice(test_errors, 20000, replace=False)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=train_errors, nbinsx=50, name="Normal training data", opacity=0.72))
    fig.add_trace(go.Histogram(x=test_errors, nbinsx=50, name="Test data", opacity=0.55))
    fig.add_vline(x=float(result["threshold"]), line_dash="dash", line_width=2, line_color="#c65c48", annotation_text="Threshold")
    fig.update_layout(
        title="Reconstruction-error distribution",
        xaxis_title="Reconstruction error",
        yaxis_title="Rows",
        barmode="overlay",
        height=330,
        margin=dict(l=18, r=18, t=55, b=35),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False, "responsive": True})


def _dashboard_result_context() -> dict[str, Any]:
    result = LAST_RESULT
    persisted = load_saved_metrics() if result is None else None
    source = CURRENT_SOURCE_DF
    try:
        source = source if source is not None else load_default_dataset()
    except (OSError, pd.errors.ParserError, FileNotFoundError):
        source = pd.DataFrame()
    label_col = detect_label_column(source) if not source.empty else None
    numeric_features = [
        str(c) for c in source.select_dtypes(include=[np.number, "bool"]).columns if c != label_col
    ] if not source.empty else []
    artifacts = [
        name for name in ["autoencoder.pt", "scaler.pkl", "model_metadata.json", "metrics.json", "loss_curve.png", "error_distribution.png"]
        if (ARTIFACT_DIR / name).exists()
    ]
    active_metrics = result.get("metrics") if result is not None else (persisted or {}).get("metrics")
    history = result.get("history") if result is not None else (persisted or {}).get("history")
    return {
        "source_rows": len(source),
        "feature_count": len(numeric_features),
        "label_column": label_col or "Not detected",
        "source_name": CURRENT_SOURCE_NAME,
        "missing_count": int(source.isna().sum().sum()) if not source.empty else 0,
        "duplicate_count": int(source.duplicated().sum()) if not source.empty else 0,
        "model_available": get_bundle() is not None,
        "artifacts": artifacts,
        "metrics": active_metrics,
        "threshold": result.get("threshold") if result is not None else (persisted or {}).get("threshold"),
        "threshold_description": result.get("threshold_description") if result is not None else (persisted or {}).get("threshold_description"),
        "train_samples": result.get("train_samples") if result is not None else (persisted or {}).get("train_samples"),
        "test_samples": result.get("test_samples") if result is not None else (persisted or {}).get("test_samples"),
        "anomalies_detected": result.get("anomalies_detected") if result is not None else (persisted or {}).get("anomalies_detected"),
        "device": result.get("device") if result is not None else (persisted or {}).get("device"),
        "loss_chart": build_loss_chart(history),
        "error_chart": build_error_chart(result),
        "last_result_available": result is not None,
        "prediction_preview": (
            result["test_results"].sort_values("reconstruction_error", ascending=False).head(25)
            .replace({np.nan: None}).to_dict(orient="records") if result is not None else []
        ),
        "prediction_columns": (list(result["test_results"].columns) if result is not None else []),
    }


def parse_float_form(name: str, default: float, minimum: float, maximum: float) -> float:
    try:
        value = float(request.form.get(name, default))
    except (TypeError, ValueError):
        raise ValueError(f"'{name.replace('_', ' ').title()}' must be a number.") from None
    if not minimum <= value <= maximum:
        raise ValueError(f"'{name.replace('_', ' ').title()}' must be between {minimum} and {maximum}.")
    return value


@app.context_processor
def inject_template_helpers() -> dict[str, Any]:
    return {"fmt_int": fmt_int, "normalised_label": normalised_label, "model_status": get_bundle() is not None}


@app.route("/")
def dashboard():
    try:
        context = _dashboard_result_context()
    except Exception as exc:  # Surface a useful page-level message instead of crashing on the homepage.
        flash(f"Dashboard data could not be loaded: {exc}", "error")
        context = {
            "source_rows": 0, "feature_count": 0, "label_column": "Not detected",
            "source_name": CURRENT_SOURCE_NAME, "missing_count": 0, "duplicate_count": 0,
            "model_available": get_bundle() is not None, "artifacts": [], "metrics": None,
            "threshold": None, "threshold_description": None, "train_samples": None,
            "test_samples": None, "anomalies_detected": None, "device": None,
            "loss_chart": None, "error_chart": None, "last_result_available": False,
            "prediction_preview": [], "prediction_columns": [],
        }
    return render_template("dashboard.html", **context)


@app.route("/train", methods=["POST"])
def train():
    global CURRENT_SOURCE_DF, CURRENT_SOURCE_NAME, CURRENT_BUNDLE, LAST_RESULT, LAST_TRAINING_SUMMARY

    uploaded = request.files.get("dataset_file")
    source_df: pd.DataFrame
    source_name: str
    try:
        if uploaded and uploaded.filename:
            filename = secure_filename(uploaded.filename)
            if not filename.lower().endswith(".csv"):
                raise ValueError("Upload a CSV file only.")
            source_df = pd.read_csv(uploaded.stream)
            source_name = filename
        else:
            source_df = CURRENT_SOURCE_DF if CURRENT_SOURCE_DF is not None else load_default_dataset()
            source_name = CURRENT_SOURCE_NAME

        max_rows_raw = request.form.get("max_rows", "20000")
        max_rows = None if max_rows_raw == "all" else int(max_rows_raw)
        epochs = int(request.form.get("epochs", 20))
        batch_size = int(request.form.get("batch_size", 256))
        latent_dim = int(request.form.get("latent_dim", 8))
        patience = int(request.form.get("patience", 5))
        learning_rate = parse_float_form("learning_rate", 0.001, 0.0001, 0.003)
        threshold_method = request.form.get("threshold_method", "percentile")
        if threshold_method not in {"percentile", "mean_std"}:
            raise ValueError("Select a valid threshold method.")
        threshold_min, threshold_max = (90.0, 99.9) if threshold_method == "percentile" else (1.0, 5.0)
        threshold_default = 99.0 if threshold_method == "percentile" else 3.0
        threshold_setting = parse_float_form("threshold_setting", threshold_default, threshold_min, threshold_max)

        if max_rows is not None and max_rows not in {20_000, 50_000, 100_000, 250_000}:
            raise ValueError("Select a valid row limit.")
        if epochs not in {5, 10, 15, 20, 25, 30, 40, 50, 60, 70, 80}:
            raise ValueError("Epochs must be one of the available options.")
        if batch_size not in {32, 64, 128, 256, 512, 1024}:
            raise ValueError("Select a valid batch size.")
        if latent_dim not in {2, 4, 8, 16, 32}:
            raise ValueError("Select a valid latent dimension.")
        if not 3 <= patience <= 15:
            raise ValueError("Early-stopping patience must be between 3 and 15.")

        prepared = prepare_dataset(source_df, max_rows=max_rows, random_state=42)
        result = train_pipeline(
            prepared,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            latent_dim=latent_dim,
            patience=patience,
            threshold_method=threshold_method,
            threshold_setting=threshold_setting,
            random_state=42,
        )
        CURRENT_SOURCE_DF = source_df
        CURRENT_SOURCE_NAME = source_name
        CURRENT_BUNDLE = {
            key: result[key]
            for key in [
                "model", "scaler", "threshold", "threshold_method", "threshold_setting",
                "threshold_description", "feature_columns", "label_column", "input_defaults",
                "hidden_dims", "latent_dim",
            ]
        }
        LAST_RESULT = result
        LAST_TRAINING_SUMMARY = {
            "used_rows": prepared["used_rows"],
            "valid_source_rows": prepared["valid_source_rows"],
            "train_rows": prepared["train_rows"],
            "test_rows": prepared["test_rows"],
            "label_distribution": prepared["label_distribution"],
        }
        flash(
            f"Training completed. Used {fmt_int(prepared['used_rows'])} valid rows; threshold = {result['threshold']:.6g}.",
            "success",
        )
    except (ValueError, OSError, pd.errors.ParserError, KeyError, RuntimeError) as exc:
        flash(f"Training failed: {exc}", "error")
    except Exception as exc:
        app.logger.exception("Unexpected training error")
        flash(f"An unexpected error stopped training: {exc}", "error")

    return redirect(url_for("dashboard"))


@app.route("/single", methods=["GET", "POST"])
def single_transaction():
    bundle = get_bundle()
    form_values: dict[str, str] = {}
    prediction: dict[str, Any] | None = None
    if bundle is not None:
        feature_columns = bundle["feature_columns"]
        defaults = bundle.get("input_defaults", [0.0] * len(feature_columns))
        form_values = {str(column): str(float(defaults[i])) for i, column in enumerate(feature_columns)}
    else:
        feature_columns = []

    if request.method == "POST":
        if bundle is None:
            flash("Train a model or load saved artifacts before scoring a transaction.", "error")
        else:
            try:
                row: dict[str, float] = {}
                for column in feature_columns:
                    raw = request.form.get(column, "").strip()
                    if not raw:
                        raise ValueError(f"Enter a value for '{column}'.")
                    try:
                        row[column] = float(raw)
                    except ValueError:
                        raise ValueError(f"'{column}' must be numeric.") from None
                    if not np.isfinite(row[column]):
                        raise ValueError(f"'{column}' must be a finite number.")
                form_values = {key: str(value) for key, value in row.items()}
                scored = score_dataframe(bundle, pd.DataFrame([row]))
                prediction = {
                    "label": str(scored.iloc[0]["prediction"]),
                    "is_anomaly": bool(scored.iloc[0]["predicted_anomaly"]),
                    "error": float(scored.iloc[0]["reconstruction_error"]),
                    "threshold": float(bundle["threshold"]),
                }
            except (ValueError, KeyError) as exc:
                flash(f"Could not score transaction: {exc}", "error")
    return render_template(
        "single_transaction.html",
        bundle=bundle,
        feature_columns=feature_columns,
        form_values=form_values,
        prediction=prediction,
    )


@app.route("/batch", methods=["GET", "POST"])
def batch_scoring():
    bundle = get_bundle()
    result: dict[str, Any] | None = None
    if request.method == "POST":
        if bundle is None:
            flash("Train a model or load saved artifacts before batch scoring.", "error")
        else:
            uploaded = request.files.get("batch_file")
            if not uploaded or not uploaded.filename:
                flash("Choose a CSV file to score.", "error")
            elif not secure_filename(uploaded.filename).lower().endswith(".csv"):
                flash("Upload a CSV file only.", "error")
            else:
                try:
                    batch_df = pd.read_csv(uploaded.stream)
                    scored = score_dataframe(bundle, batch_df)
                    flagged = int(scored["predicted_anomaly"].sum())
                    BATCH_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
                    export_name = f"scored_transactions_{uuid.uuid4().hex[:12]}.csv"
                    scored.to_csv(BATCH_RESULTS_DIR / export_name, index=False)
                    result = {
                        "filename": secure_filename(uploaded.filename),
                        "rows": len(scored),
                        "flagged": flagged,
                        "rate": flagged / max(len(scored), 1),
                        "preview": scored.sort_values("reconstruction_error", ascending=False).head(100).replace({np.nan: None}).to_dict(orient="records"),
                        "columns": [str(c) for c in scored.columns],
                        "download_name": export_name,
                    }
                except (ValueError, OSError, pd.errors.ParserError, KeyError) as exc:
                    flash(f"Could not score the uploaded CSV: {exc}", "error")
                except Exception as exc:
                    app.logger.exception("Unexpected batch scoring error")
                    flash(f"An unexpected error stopped batch scoring: {exc}", "error")
    return render_template("batch_scoring.html", bundle=bundle, result=result)


@app.route("/download/<filename>")
def download_batch(filename: str):
    # Only files generated by this app in the batch-results directory are downloadable.
    safe_name = secure_filename(filename)
    if safe_name != filename or not filename.startswith("scored_transactions_") or not filename.endswith(".csv"):
        abort(404)
    path = BATCH_RESULTS_DIR / filename
    if not path.is_file():
        abort(404)
    return send_from_directory(BATCH_RESULTS_DIR, filename, as_attachment=True, download_name="scored_transactions.csv")


@app.route("/dataset")
def dataset_explorer():
    try:
        source_df = CURRENT_SOURCE_DF if CURRENT_SOURCE_DF is not None else load_default_dataset()
        label_col = detect_label_column(source_df)
        numeric_features = [str(c) for c in source_df.select_dtypes(include=[np.number, "bool"]).columns if c != label_col]
        selected_feature = request.args.get("feature", numeric_features[0] if numeric_features else "")
        if selected_feature not in numeric_features and numeric_features:
            selected_feature = numeric_features[0]
        preview_columns = [str(c) for c in source_df.columns]
        preview_rows = source_df.head(15).replace({np.nan: None}).to_dict(orient="records")
        column_summary = [
            {
                "column": str(column),
                "dtype": str(source_df[column].dtype),
                "missing": int(source_df[column].isna().sum()),
                "unique": int(source_df[column].nunique(dropna=True)),
            }
            for column in source_df.columns
        ]
        label_counts: list[dict[str, Any]] = []
        label_chart = None
        if label_col:
            counts = source_df[label_col].value_counts(dropna=False).rename_axis("Label").reset_index(name="Rows")
            counts["Label"] = counts["Label"].astype(str)
            label_counts = counts.to_dict(orient="records")
            fig = px.bar(counts, x="Label", y="Rows", color="Label", text_auto=".2s", title="Label distribution")
            fig.update_layout(showlegend=False, height=310, margin=dict(l=12, r=12, t=50, b=35), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            label_chart = fig.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False, "responsive": True})
        feature_chart = None
        if selected_feature:
            chart_df = source_df[[selected_feature]].dropna()
            if len(chart_df) > 20000:
                chart_df = chart_df.sample(20000, random_state=42)
            fig = px.histogram(chart_df, x=selected_feature, nbins=40, title=f"Distribution: {selected_feature}")
            fig.update_layout(height=310, margin=dict(l=12, r=12, t=50, b=35), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            feature_chart = fig.to_html(full_html=False, include_plotlyjs=False, config={"displayModeBar": False, "responsive": True})
        return render_template(
            "dataset_explorer.html",
            source_df=source_df,
            source_name=CURRENT_SOURCE_NAME,
            row_count=len(source_df),
            column_count=len(source_df.columns),
            missing_count=int(source_df.isna().sum().sum()),
            duplicate_count=int(source_df.duplicated().sum()),
            label_col=label_col,
            preview_columns=preview_columns,
            preview_rows=preview_rows,
            column_summary=column_summary,
            label_counts=label_counts,
            label_chart=label_chart,
            numeric_features=numeric_features,
            selected_feature=selected_feature,
            feature_chart=feature_chart,
        )
    except (OSError, pd.errors.ParserError, FileNotFoundError) as exc:
        flash(f"Could not load the dataset: {exc}", "error")
        return render_template("dataset_explorer.html", source_df=pd.DataFrame(), source_name=CURRENT_SOURCE_NAME,
                               row_count=0, column_count=0, missing_count=0, duplicate_count=0,
                               label_col=None, preview_columns=[], preview_rows=[], column_summary=[],
                               label_counts=[], label_chart=None, numeric_features=[], selected_feature="", feature_chart=None)


@app.route("/artifacts/<filename>")
def artifact_image(filename: str):
    if filename not in ALLOWED_ARTIFACT_IMAGES:
        abort(404)
    if not (ARTIFACT_DIR / filename).is_file():
        abort(404)
    return send_from_directory(ARTIFACT_DIR, filename)


@app.errorhandler(413)
def file_too_large(_error):
    flash("The uploaded file exceeds the 300 MB limit.", "error")
    return redirect(url_for("dashboard"))


if __name__ == "__main__":
    # Debug mode is convenient locally, but disable it for any deployed instance.
    app.run(host="127.0.0.1", port=5000, debug=True)
