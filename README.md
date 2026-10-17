Customer Churn Prediction — Machine Learning Project

An end-to-end machine learning project that predicts customer churn using Python, pandas, and scikit-learn. It compares Logistic Regression and Random Forest and provides a Streamlit dashboard for predictions and model evaluation.

Project Features

- Customer churn data loading and preprocessing
- Missing-value handling and categorical encoding
- Logistic Regression and Random Forest models
- Model evaluation using accuracy, precision, recall, F1-score, and ROC-AUC
- Streamlit dashboard for predictions and evaluation reports

Project Structure

customer_churn_prediction_project/
├── app.py
├── requirements.txt
├── data/
│   └── README.md
├── reports/
│   ├── metrics.json
│   └── test_predictions.csv
└── src/
    ├── common.py
    ├── download_data.py
    └── train.py

Installation (Windows)

Open PowerShell in the project folder and run:

py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python src/download_data.py

Train the Model

python src/train.py

The training script preprocesses the dataset, trains the models, evaluates their performance, and saves the generated reports and model artifacts.

Run the Dashboard

streamlit run app.py

Open the local URL shown in the terminal.

Evaluation

The project evaluates models using accuracy, precision, recall, F1-score, and ROC-AUC. Actual evaluation results should be taken from the generated report files rather than assumed.

Dataset

This project uses the IBM Telco Customer Churn dataset. See "data/README.md" for dataset information and download instructions.

Disclaimer

This is an educational portfolio project. Model predictions are estimates and should not be treated as certain outcomes.