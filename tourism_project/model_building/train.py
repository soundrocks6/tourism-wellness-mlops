# -----------------------------------------------------------------------------
# train.py
# Model training and registration step with experiment tracking.
# - Loads the train/test splits produced by the data preparation job
# - Defines an XGBoost model and a hyperparameter grid, and tunes it
# - Logs all tuned parameters and evaluation metrics to MLflow
# - Evaluates the best model on the test set
# - Saves the best model into tourism_project/deployment/ so the workflow
#   can commit it back to the repository
# -----------------------------------------------------------------------------

import joblib
import mlflow
import pandas as pd
import requests
import xgboost as xgb
from sklearn.compose import make_column_transformer
from sklearn.metrics import (accuracy_score, classification_report, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

MODEL_PATH = "tourism_project/deployment/model.joblib"

# ---- Point MLflow at the tracking server started by the workflow.
#      If no server is running (e.g. local execution), fall back to a
#      local file-based store so the script still works everywhere. ----
TRACKING_URI = "http://localhost:5000"
try:
    requests.get(TRACKING_URI, timeout=3)
    mlflow.set_tracking_uri(TRACKING_URI)
    print(f"MLflow tracking server found at {TRACKING_URI}")
except requests.exceptions.RequestException:
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    print("No MLflow server found; using local store sqlite:///mlflow.db")

mlflow.set_experiment("tourism-wellness-package")

def main():
    # ---- Load the splits passed from the data preparation job ----
    Xtrain = pd.read_csv("Xtrain.csv")
    Xtest = pd.read_csv("Xtest.csv")
    ytrain = pd.read_csv("ytrain.csv").squeeze()
    ytest = pd.read_csv("ytest.csv").squeeze()
    print(f"Train: {Xtrain.shape} | Test: {Xtest.shape}")

    # ---- Identify feature types for preprocessing ----
    numeric_features = Xtrain.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = Xtrain.select_dtypes(exclude=["int64", "float64"]).columns.tolist()
    print("Numeric features    :", numeric_features)
    print("Categorical features:", categorical_features)

    # ---- Preprocessing: scale numeric, one-hot encode categorical ----
    preprocessor = make_column_transformer(
        (StandardScaler(), numeric_features),
        (OneHotEncoder(handle_unknown="ignore"), categorical_features),
    )

    # ---- Handle class imbalance (~81% non-buyers vs ~19% buyers) ----
    scale_pos_weight = (ytrain == 0).sum() / (ytrain == 1).sum()
    print(f"scale_pos_weight = {scale_pos_weight:.2f}")

    # ---- Define the model ----
    model = xgb.XGBClassifier(
        random_state=42,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
    )
    pipeline = make_pipeline(preprocessor, model)

    # ---- Define the hyperparameter grid to tune ----
    param_grid = {
        "xgbclassifier__n_estimators": [100, 200],
        "xgbclassifier__max_depth": [3, 5, 7],
        "xgbclassifier__learning_rate": [0.05, 0.1],
    }

    with mlflow.start_run(run_name="xgboost-gridsearch"):
        # ---- Tune the model with the defined parameters ----
        grid = GridSearchCV(
            pipeline, param_grid, cv=5, scoring="f1", n_jobs=-1, verbose=1
        )
        grid.fit(Xtrain, ytrain)

        # ---- Log every tuned parameter combination as a nested run ----
        results = grid.cv_results_
        for i, params in enumerate(results["params"]):
            with mlflow.start_run(nested=True, run_name=f"candidate_{i}"):
                mlflow.log_params(params)
                mlflow.log_metric("mean_cv_f1", results["mean_test_score"][i])
                mlflow.log_metric("std_cv_f1", results["std_test_score"][i])

        # ---- Log the best parameters found ----
        print("\nBest parameters:", grid.best_params_)
        print(f"Best CV F1 score: {grid.best_score_:.4f}")
        mlflow.log_params(grid.best_params_)
        mlflow.log_metric("best_cv_f1", grid.best_score_)

        # ---- Evaluate the best model on train and test sets ----
        best_model = grid.best_estimator_
        for split_name, X_, y_ in [("train", Xtrain, ytrain), ("test", Xtest, ytest)]:
            preds = best_model.predict(X_)
            proba = best_model.predict_proba(X_)[:, 1]
            metrics = {
                f"{split_name}_accuracy": accuracy_score(y_, preds),
                f"{split_name}_precision": precision_score(y_, preds),
                f"{split_name}_recall": recall_score(y_, preds),
                f"{split_name}_f1": f1_score(y_, preds),
                f"{split_name}_roc_auc": roc_auc_score(y_, proba),
            }
            mlflow.log_metrics(metrics)
            print(f"\n--- {split_name.upper()} performance ---")
            for k, v in metrics.items():
                print(f"{k:>18}: {v:.4f}")

        print("\nClassification report (test set):")
        print(classification_report(ytest, best_model.predict(Xtest)))

        # ---- Save the best model so the pipeline can commit it to the repo ----
        joblib.dump(best_model, MODEL_PATH)
        mlflow.log_artifact(MODEL_PATH)
        print(f"Best model saved to {MODEL_PATH}")

if __name__ == "__main__":
    main()
