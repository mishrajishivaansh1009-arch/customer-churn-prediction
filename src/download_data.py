from pathlib import Path
from urllib.request import urlretrieve
import sys

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
DEST = DATA_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"

URLS = [
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv",
    "https://raw.githubusercontent.com/IBM/Project-599-Telco-Customer-Churn-Analysis/master/Telco-Customer-Churn.csv",
]

def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if DEST.exists():
        print(f"Dataset already exists: {DEST}")
        return
    errors = []
    for url in URLS:
        try:
            print(f"Trying: {url}")
            urlretrieve(url, DEST)
            # Basic sanity check prevents saving an HTML error page as CSV.
            with DEST.open("r", encoding="utf-8-sig") as f:
                header = f.readline().lower()
            if "customerid" in header and "churn" in header:
                print(f"Downloaded dataset to: {DEST}")
                return
            DEST.unlink(missing_ok=True)
            errors.append(f"{url}: downloaded file did not look like expected CSV")
        except Exception as exc:
            DEST.unlink(missing_ok=True)
            errors.append(f"{url}: {exc}")
    print("Automatic download failed.")
    for error in errors:
        print(" -", error)
    print("Manually download the IBM Telco Customer Churn CSV and save it as:")
    print(DEST)
    sys.exit(1)

if __name__ == "__main__":
    main()
