# Customer Churn Prediction — Placement-Ready ML Project

A beginner-friendly end-to-end classification project. It loads customer subscription data, cleans it, compares Logistic Regression and Random Forest, selects the best model by held-out test ROC-AUC, saves metrics, and provides a Streamlit dashboard.

## 1. Project structure

```text
customer_churn_prediction_project/
├── app.py
├── requirements.txt
├── data/
│   └── README.md
├── models/                 # trained model created after training
├── reports/                # metrics, predictions, charts created after training
└── src/
    ├── common.py
    ├── download_data.py
    └── train.py
```

## 2. Setup (Windows / PowerShell)

Open terminal in this folder:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python src/download_data.py
```

If PowerShell blocks activation, use `Set-ExecutionPolicy -Scope Process Bypass` for that terminal only, then activate again. Alternatively use `.venv\Scripts\python.exe` directly.

If automatic download fails, get the IBM Telco Customer Churn CSV from the dataset source/repository and save it as `data/WA_Fn-UseC_-Telco-Customer-Churn.csv`. Confirm it contains a `Churn` column.

## 3. Train and evaluate

```powershell
python src/train.py
```

This performs:
1. Basic data checks and duplicate removal.
2. Target mapping (`No` → 0, `Yes` → 1).
3. Stratified train/test split (80/20).
4. Missing-value imputation and scaling for numeric fields.
5. One-hot encoding for categorical fields.
6. Training of Logistic Regression and Random Forest.
7. Test evaluation using accuracy, precision, recall, F1 and ROC-AUC.
8. Selection by test ROC-AUC and saving the pipeline, metrics, test predictions, confusion matrix, and comparison chart.

**Important evaluation note:** selecting a model based on the test set makes the test set part of model selection. For a stronger production study, add cross-validation on the training set and keep a final untouched test set. This educational baseline makes the workflow easy to inspect.

## 4. Launch dashboard

```powershell
streamlit run app.py
```

Streamlit prints a local URL, usually `http://localhost:8501`. The dashboard contains overview, one-customer prediction, and model evaluation tabs.

## 5. Explain the project in an interview

- **Problem:** customer churn is when a customer stops using a service.
- **Target:** `Churn` (Yes/No).
- **Task:** binary classification.
- **Preprocessing:** median imputation and scaling for numeric fields; most-frequent imputation and one-hot encoding for categorical fields.
- **Models:** Logistic Regression (interpretable baseline) and Random Forest (non-linear ensemble).
- **Metrics:** precision, recall, F1 and ROC-AUC alongside accuracy. Recall matters when missing a likely churner is costly; precision matters when retention outreach is expensive.
- **Business use:** prioritise customer-care review and retention offers; do not treat a model score as certainty or use it without human oversight.

## 6. Before submission

- Run the project on your machine.
- Copy actual metrics from `reports/metrics.json` into the report.
- Add your real name, roll number, supervisor, dates, and institute details.
- Replace all placeholders only with verified information.
- Do not claim internship work, deployment, or results that did not occur.
