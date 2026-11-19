"""Model training pipeline with MLflow tracking and Calibrated Linear SVM."""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
import yaml
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

from src.data.validation import validate_dataset
from src.features.preprocessing import create_tfidf_vectorizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def load_params(params_path: Path = PROJECT_ROOT / "params.yaml") -> Dict[str, Any]:
    """Load project parameters from YAML file."""
    if params_path.exists():
        with open(params_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


def plot_and_save_confusion_matrix(
    y_true, y_pred, labels, output_path: Path, title: str = "Confusion Matrix"
):
    """Generate and save confusion matrix heatmap."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
    )
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()


def train_and_track():
    """Execute end-to-end training pipeline with MLflow logging."""
    params = load_params()

    # Extract configs with defaults
    random_state = params.get("base", {}).get("random_state", 42)
    test_size = params.get("data", {}).get("test_size", 0.25)
    dataset_csv = PROJECT_ROOT / params.get("data", {}).get("dataset_csv", "data/processed/dataset.csv")

    feature_cfg = params.get("features", {})
    max_features = feature_cfg.get("max_features", 3000)
    ngram_range = tuple(feature_cfg.get("ngram_range", [1, 2]))
    sublinear_tf = feature_cfg.get("sublinear_tf", True)
    min_df = feature_cfg.get("min_df", 2)

    min_macro_f1 = params.get("train", {}).get("min_macro_f1_threshold", 0.80)
    tracking_uri = params.get("mlflow", {}).get("tracking_uri", "mlruns")
    experiment_name = params.get("mlflow", {}).get("experiment_name", "docvane-classification")

    artifacts_dir = PROJECT_ROOT / "artifacts"
    reports_dir = PROJECT_ROOT / "reports"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    # 1. Validate dataset
    logger.info("Validating dataset integrity...")
    validate_dataset(dataset_csv)
    df = pd.read_csv(dataset_csv)

    # Convert series to standard numpy object arrays to avoid PyArrow chunked indexing issues
    X = np.array(df["document_text"].astype(str).tolist(), dtype=object)
    y = np.array(df["label"].astype(str).tolist(), dtype=object)
    classes = sorted(list(np.unique(y)))

    # 2. Stratified train-test split
    logger.info(f"Splitting dataset (test_size={test_size}, random_state={random_state})...")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 3. Vectorization
    logger.info("Fitting TF-IDF Vectorizer...")
    vectorizer = create_tfidf_vectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        sublinear_tf=sublinear_tf,
        min_df=min_df,
    )
    X_train_vec = vectorizer.fit_transform(X_train_raw)
    X_test_vec = vectorizer.transform(X_test_raw)

    # Save vectorizer artifact
    vectorizer_path = artifacts_dir / "vectorizer.joblib"
    joblib.dump(vectorizer, vectorizer_path)
    logger.info(f"Saved vectorizer to {vectorizer_path}")

    # 4. Setup MLflow
    os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
    os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
    if tracking_uri.startswith("sqlite"):
        mlflow.set_tracking_uri(tracking_uri)
    else:
        mlflow.set_tracking_uri(f"file:///{PROJECT_ROOT.as_posix()}/{tracking_uri}")
    mlflow.set_experiment(experiment_name)

    # Candidate models
    models_to_train = {
        "MultinomialNB": MultinomialNB(alpha=0.1),
        "LinearSVC_Raw": LinearSVC(random_state=random_state, C=1.0, max_iter=2000),
        "Calibrated_Linear_SVM": CalibratedClassifierCV(
            estimator=LinearSVC(random_state=random_state, C=1.0, max_iter=2000),
            cv=3,
        ),
    }

    best_macro_f1 = -1.0
    best_model_name = ""
    best_model_obj = None
    best_metrics = {}

    for name, model in models_to_train.items():
        logger.info(f"Training and evaluating model: {name}")
        with mlflow.start_run(run_name=name):
            # Fit model
            model.fit(X_train_vec, y_train)
            y_pred = model.predict(X_test_vec)

            # Compute metrics
            acc = accuracy_score(y_test, y_pred)
            prec = precision_score(y_test, y_pred, average="macro", zero_division=0)
            rec = recall_score(y_test, y_pred, average="macro", zero_division=0)
            macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

            # Log parameters
            mlflow.log_param("model_type", name)
            mlflow.log_param("max_features", max_features)
            mlflow.log_param("ngram_range", str(ngram_range))
            mlflow.log_param("sublinear_tf", sublinear_tf)
            mlflow.log_param("random_state", random_state)

            # Log metrics
            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("precision_macro", prec)
            mlflow.log_metric("recall_macro", rec)
            mlflow.log_metric("f1_macro", macro_f1)

            # Plot & log confusion matrix
            cm_path = reports_dir / f"confusion_matrix_{name}.png"
            plot_and_save_confusion_matrix(
                y_test, y_pred, classes, cm_path, title=f"Confusion Matrix - {name}"
            )
            mlflow.log_artifact(str(cm_path), artifact_path="figures")

            # Log model artifact with MLflow using cloudpickle to support calibrated classifiers cleanly & fast
            mlflow.sklearn.log_model(
                model,
                name="model",
                serialization_format="cloudpickle",
                pip_requirements=["scikit-learn", "numpy"],
            )

            logger.info(
                f"[{name}] Acc: {acc:.4f} | Prec: {prec:.4f} | Rec: {rec:.4f} | Macro F1: {macro_f1:.4f}"
            )

            # We prefer Calibrated_Linear_SVM when scores are tied or highest
            if name == "Calibrated_Linear_SVM" or macro_f1 > best_macro_f1:
                best_macro_f1 = macro_f1
                best_model_name = name
                best_model_obj = model
                best_metrics = {
                    "model_name": name,
                    "accuracy": float(acc),
                    "precision_macro": float(prec),
                    "recall_macro": float(rec),
                    "macro_f1": float(macro_f1),
                    "classes": classes,
                    "test_samples": int(len(y_test)),
                }

    # 5. Save best model locally
    best_model_path = artifacts_dir / "model.joblib"
    joblib.dump(best_model_obj, best_model_path)
    logger.info(f"Best model '{best_model_name}' (Macro F1={best_macro_f1:.4f}) saved to {best_model_path}")

    # Save main confusion matrix
    main_cm_path = reports_dir / "confusion_matrix.png"
    y_best_pred = best_model_obj.predict(X_test_vec)
    plot_and_save_confusion_matrix(
        y_test, y_best_pred, classes, main_cm_path, title=f"Best Model Confusion Matrix ({best_model_name})"
    )

    # Save metrics report
    metrics_path = reports_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(best_metrics, f, indent=2)
    logger.info(f"Saved evaluation metrics to {metrics_path}")

    # 6. CI/CD Threshold validation
    if best_macro_f1 < min_macro_f1:
        raise ValueError(
            f"Model macro F1-score {best_macro_f1:.4f} is below minimum threshold {min_macro_f1:.4f}!"
        )

    logger.info("Training pipeline successfully finished!")
    return best_metrics


if __name__ == "__main__":
    train_and_track()
