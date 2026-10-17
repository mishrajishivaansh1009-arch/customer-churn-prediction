# Dataset

The project expects the public IBM Telco Customer Churn CSV at:
`data/WA_Fn-UseC_-Telco-Customer-Churn.csv`

Run `python src/download_data.py` from the project root to attempt download from the public IBM sample repository.
If that URL is unavailable, manually download the IBM Telco Customer Churn dataset and save it at the path above.

The dataset typically includes customer demographics, subscribed services, contract/payment information, monthly charges, tenure, and a `Churn` target. Always inspect the actual CSV columns before modelling. This project does not bundle the dataset.
