from pathlib import Path
import json
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from common import ROOT, load_data, TARGET, MODEL_PATH, METRICS_PATH, PREDICTIONS_PATH

RANDOM_STATE = 42

def main():
    (ROOT / "models").mkdir(exist_ok=True)
    (ROOT / "reports").mkdir(exist_ok=True)
    df = load_data()
    X = df.drop(columns=[TARGET])
    y = df[TARGET].map({"No": 0, "Yes": 1})

    # Keep a held-out test set; all preprocessing is fitted only on training data.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    numeric = X_train.select_dtypes(include=["number"]).columns.tolist()
    categorical = [c for c in X_train.columns if c not in numeric]

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocess = ColumnTransformer([
        ("num", numeric_pipe, numeric),
        ("cat", categorical_pipe, categorical)
    ])

    candidates = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1
        )
    }

    results = []
    fitted = {}
    for name, classifier in candidates.items():
        pipeline = Pipeline([("preprocess", preprocess), ("model", classifier)])
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        proba = pipeline.predict_proba(X_test)[:, 1]
        row = {
            "model": name,
            "accuracy": float(accuracy_score(y_test, pred)),
            "precision": float(precision_score(y_test, pred, zero_division=0)),
            "recall": float(recall_score(y_test, pred, zero_division=0)),
            "f1": float(f1_score(y_test, pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_test, proba))
        }
        results.append(row)
        fitted[name] = (pipeline, pred, proba)
        print(f"\n{name}")
        print(json.dumps(row, indent=2))
        print(classification_report(y_test, pred, target_names=["No Churn", "Churn"], zero_division=0))

    # Select by ROC-AUC; do not assume one model is always best.
    best_row = max(results, key=lambda x: x["roc_auc"])
    best_name = best_row["model"]
    best_pipeline, best_pred, best_proba = fitted[best_name]
    joblib.dump(best_pipeline, MODEL_PATH)

    metrics = {
        "dataset_rows_after_basic_cleaning": int(len(df)),
        "feature_count_before_encoding": int(X.shape[1]),
        "target": TARGET,
        "positive_class": "Yes",
        "test_size": 0.20,
        "random_state": RANDOM_STATE,
        "selection_metric": "roc_auc",
        "selected_model": best_name,
        "model_comparison": results,
        "selected_model_confusion_matrix": confusion_matrix(y_test, best_pred).tolist(),
        "selected_model_classification_report": classification_report(
            y_test, best_pred, target_names=["No Churn", "Churn"], output_dict=True, zero_division=0
        )
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    out = X_test.copy()
    out["actual_churn"] = y_test.map({0: "No", 1: "Yes"}).values
    out["predicted_churn"] = pd.Series(best_pred, index=X_test.index).map({0: "No", 1: "Yes"}).values
    out["churn_probability"] = best_proba
    out.to_csv(PREDICTIONS_PATH, index=False)

    cm = confusion_matrix(y_test, best_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No Churn", "Churn"], yticklabels=["No Churn", "Churn"])
    plt.xlabel("Predicted label")
    plt.ylabel("Actual label")
    plt.title(f"Confusion Matrix — {best_name}")
    plt.tight_layout()
    plt.savefig(ROOT / "reports" / "confusion_matrix.png", dpi=160)
    plt.close()

    comparison = pd.DataFrame(results).set_index("model")
    comparison.plot(kind="bar", figsize=(10, 5), ylim=(0, 1), rot=0)
    plt.title("Held-out Test Metrics by Model")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig(ROOT / "reports" / "model_comparison.png", dpi=160)
    plt.close()

    print(f"\nSelected model: {best_name}")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")
    print(f"Saved test predictions: {PREDICTIONS_PATH}")
    print("Use the printed and saved metrics in the report; do not report unexecuted results.")

if __name__ == "__main__":
    main()
