
import json
import os
from pathlib import Path

import joblib
import pandas as pd
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    mlflow = None
    MLFLOW_AVAILABLE = False

ROOT = Path(__file__).resolve().parents[2]

# Create deployment directory at the start of training
DEPLOYMENT_DIR = ROOT / "tourism_project" / "deployment"
DEPLOYMENT_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = DEPLOYMENT_DIR / "best_tourism_package_model.joblib"
METRICS_PATH = DEPLOYMENT_DIR / "model_metrics.json"
PARAMS_PATH = DEPLOYMENT_DIR / "best_model_params.json"
TUNING_PATH = DEPLOYMENT_DIR / "tuning_results.csv"


def load_splits():
    Xtrain = pd.read_csv(ROOT / "Xtrain.csv")
    Xtest = pd.read_csv(ROOT / "Xtest.csv")
    ytrain = pd.read_csv(ROOT / "ytrain.csv").squeeze("columns")
    ytest = pd.read_csv(ROOT / "ytest.csv").squeeze("columns")
    return Xtrain, Xtest, ytrain, ytest


def build_pipeline():
    numeric_features = [
        "Age",
        "CityTier",
        "DurationOfPitch",
        "NumberOfPersonVisiting",
        "NumberOfFollowups",
        "PreferredPropertyStar",
        "NumberOfTrips",
        "Passport",
        "PitchSatisfactionScore",
        "OwnCar",
        "NumberOfChildrenVisiting",
        "MonthlyIncome"
    ]

    categorical_features = [
        "TypeofContact",
        "Occupation",
        "Gender",
        "ProductPitched",
        "MaritalStatus",
        "Designation"
    ]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), numeric_features),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
    ])

    classifier = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        tree_method="hist"
    )

    return Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", classifier)
    ])


def main():
    Xtrain, Xtest, ytrain, ytest = load_splits()

    pipeline = build_pipeline()

    param_grid = {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [3, 5],
        "classifier__learning_rate": [0.05, 0.1],
        "classifier__subsample": [0.8],
        "classifier__colsample_bytree": [0.8],
        "classifier__scale_pos_weight": [1, 2]
    }

    if MLFLOW_AVAILABLE:
        os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
        mlflow.set_tracking_uri(
            os.getenv(
                "MLFLOW_TRACKING_URI",
                f"file://{ROOT / 'mlruns'}"
            )
        )

        mlflow.set_experiment("tourism-package-prediction")
        run_ctx = mlflow.start_run(
            run_name="xgboost-grid-search"
        )
    else:
        class DummyContext:
            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        run_ctx = DummyContext()

    with run_ctx:
        grid = GridSearchCV(
            pipeline,
            param_grid,
            scoring="f1",
            cv=3,
            n_jobs=-1,
            refit=True,
            verbose=1
        )

        grid.fit(Xtrain, ytrain)

        # Persist every tuned parameter combination
        # and its cross-validation score.
        tuning_df = pd.DataFrame(grid.cv_results_)[[
            "params",
            "mean_test_score",
            "std_test_score",
            "rank_test_score"
        ]].sort_values("rank_test_score")

        tuning_df.to_csv(TUNING_PATH, index=False)

        if MLFLOW_AVAILABLE:
            for _, row in tuning_df.iterrows():
                with mlflow.start_run(nested=True):
                    mlflow.log_params({
                        k: str(v)
                        for k, v in row["params"].items()
                    })

                    mlflow.log_metric(
                        "cv_f1",
                        float(row["mean_test_score"])
                    )

                    mlflow.log_metric(
                        "cv_f1_std",
                        float(row["std_test_score"])
                    )

        best_model = grid.best_estimator_

        pred_train = best_model.predict(Xtrain)
        pred_test = best_model.predict(Xtest)
        prob_test = best_model.predict_proba(Xtest)[:, 1]

        metrics = {
            "train_accuracy": accuracy_score(
                ytrain,
                pred_train
            ),
            "test_accuracy": accuracy_score(
                ytest,
                pred_test
            ),
            "test_precision": precision_score(
                ytest,
                pred_test,
                zero_division=0
            ),
            "test_recall": recall_score(
                ytest,
                pred_test,
                zero_division=0
            ),
            "test_f1": f1_score(
                ytest,
                pred_test,
                zero_division=0
            ),
            "test_roc_auc": roc_auc_score(
                ytest,
                prob_test
            )
        }

        if MLFLOW_AVAILABLE:
            mlflow.log_params({
                k: str(v)
                for k, v in grid.best_params_.items()
            })

            mlflow.log_metrics(metrics)

    # Save trained artifacts
    joblib.dump(best_model, MODEL_PATH)

    PARAMS_PATH.write_text(
        json.dumps(
            grid.best_params_,
            indent=2,
            default=str
        )
    )

    METRICS_PATH.write_text(
        json.dumps(
            metrics,
            indent=2
        )
    )

    print("\nBest parameters:")
    print(
        json.dumps(
            grid.best_params_,
            indent=2,
            default=str
        )
    )

    print("\nMetrics:")
    print(
        json.dumps(
            metrics,
            indent=2
        )
    )

    print("\nClassification report:")
    print(
        classification_report(
            ytest,
            pred_test,
            digits=4
        )
    )

    print(f"Tuning results saved to: {TUNING_PATH}")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"MLflow tracking enabled: {MLFLOW_AVAILABLE}")


if __name__ == "__main__":
    main()
