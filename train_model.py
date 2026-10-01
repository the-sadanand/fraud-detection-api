"""Generate data (if missing), train a RandomForest and log + register it with MLflow."""
import argparse
import sys
import time
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

from app.core.config import settings
from app.services.fraud_service import FEATURE_COLUMNS
from generate_data import DEFAULT_PATH, generate

EXPERIMENT_NAME = "fraud-detection"
MODEL_PKL = Path("data/fraud_model.pkl")


def wait_for_mlflow(client: MlflowClient, timeout: int = 90) -> None:
    deadline = time.time() + timeout
    while True:
        try:
            client.search_experiments(max_results=1)
            return
        except Exception as exc:
            if time.time() > deadline:
                raise RuntimeError(f"MLflow not reachable at {settings.mlflow_tracking_uri}: {exc}") from exc
            print("Waiting for MLflow server...", flush=True)
            time.sleep(3)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-path", type=Path, default=DEFAULT_PATH)
    parser.add_argument("--n-samples", type=int, default=20000)
    parser.add_argument("--n-estimators", type=int, default=150)
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-if-registered", action="store_true",
                        help="Do nothing if the model already has a registered version")
    args = parser.parse_args()

    mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
    client = MlflowClient()
    wait_for_mlflow(client)

    if args.skip_if_registered and client.search_model_versions(f"name='{settings.model_name}'"):
        print(f"Model '{settings.model_name}' already registered - skipping training.")
        return 0

    # 1. Data
    if not args.data_path.exists():
        args.data_path.parent.mkdir(parents=True, exist_ok=True)
        generate(args.n_samples, args.seed).to_csv(args.data_path, index=False)
        print(f"Generated synthetic data at {args.data_path}")
    df = pd.read_csv(args.data_path)
    X = df[FEATURE_COLUMNS].astype("float64")
    y = df["is_fraud"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    # 2. Train + track
    params = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "class_weight": "balanced",
        "random_state": args.seed,
        "n_jobs": -1,
    }
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="rf-fraud-baseline") as run:
        model = RandomForestClassifier(**params).fit(X_train, y_train)
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds, zero_division=0),
            "recall": recall_score(y_test, preds, zero_division=0),
            "f1_score": f1_score(y_test, preds, zero_division=0),
            "roc_auc": roc_auc_score(y_test, proba),
        }
        mlflow.log_params({**params, "test_size": args.test_size, "n_rows": len(df),
                           "features": ",".join(FEATURE_COLUMNS)})
        mlflow.log_metrics(metrics)

        # 3. Log + register in the Model Registry
        info = mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            signature=infer_signature(X_test, preds),
            input_example=X_train.head(3),
            registered_model_name=settings.model_name,
        )
        version = getattr(info, "registered_model_version", None)
        if version is None:
            version = max(int(v.version) for v in client.search_model_versions(f"name='{settings.model_name}'"))
        client.set_registered_model_alias(settings.model_name, "champion", str(version))

    MODEL_PKL.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PKL)

    print(f"Run {run.info.run_id} metrics: " + ", ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
    print(f"Registered '{settings.model_name}' version {version} (alias: champion)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
