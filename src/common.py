from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_PATH = ROOT / "models" / "churn_pipeline.joblib"
METRICS_PATH = ROOT / "reports" / "metrics.json"
PREDICTIONS_PATH = ROOT / "reports" / "test_predictions.csv"

TARGET = "Churn"

def load_data(path=DATA_PATH):
    if not Path(path).exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Run: python src/download_data.py"
        )
    df = pd.read_csv(path)
    if TARGET not in df.columns:
        raise ValueError(f"Expected target column '{TARGET}'. Found columns: {list(df.columns)}")
    df = df.copy()
    # In the standard IBM Telco dataset TotalCharges can contain blank strings.
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.drop_duplicates()
    # CustomerID is an identifier, not a useful generalizable customer behaviour feature.
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
    df[TARGET] = df[TARGET].astype(str).str.strip()
    df = df[df[TARGET].isin(["Yes", "No"])]
    return df
