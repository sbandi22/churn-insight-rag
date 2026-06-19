"""Download the Telco Customer Churn dataset.

The canonical dataset is the "Telco Customer Churn" dataset published on Kaggle:
    https://www.kaggle.com/datasets/blastchar/telco-customer-churn

Because Kaggle requires authentication, this script supports three strategies,
tried in order:

1. KAGGLE_USERNAME / KAGGLE_KEY environment variables + the kaggle CLI/library.
2. A direct URL provided via the TELCO_DATA_URL environment variable (e.g. a
   mirror you are licensed to use).
3. A synthetic fallback generator so the rest of the pipeline can run fully
   offline with no credentials. The synthetic data mimics the schema and
   approximate distributions of the real Telco dataset.

Usage:
    python data/download_telco.py
"""
from __future__ import annotations

import os
import sys
import random
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent
TARGET = DATA_DIR / "telco_churn.csv"

KAGGLE_DATASET = "blastchar/telco-customer-churn"
KAGGLE_FILE = "WA_Fn-UseC_-Telco-Customer-Churn.csv"


def _try_kaggle() -> bool:
    """Download via the Kaggle API if credentials are present."""
    if not (os.getenv("KAGGLE_USERNAME") and os.getenv("KAGGLE_KEY")):
        return False
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        print(f"Downloading {KAGGLE_DATASET} from Kaggle...")
        api.dataset_download_files(KAGGLE_DATASET, path=str(DATA_DIR), unzip=True)
        src = DATA_DIR / KAGGLE_FILE
        if src.exists():
            src.replace(TARGET)
        return TARGET.exists()
    except Exception as exc:  # pragma: no cover
        print(f"Kaggle download failed: {exc}")
        return False


def _try_url() -> bool:
    """Download from a user-provided mirror URL."""
    url = os.getenv("TELCO_DATA_URL")
    if not url:
        return False
    try:
        import urllib.request

        print("Downloading Telco dataset from TELCO_DATA_URL...")
        urllib.request.urlretrieve(url, TARGET)
        return TARGET.exists()
    except Exception as exc:  # pragma: no cover
        print(f"URL download failed: {exc}")
        return False


def _synthesize(n: int = 400, seed: int = 42) -> None:
    """Generate a synthetic Telco-like dataset matching the real schema."""
    import csv

    random.seed(seed)
    print(f"Generating {n} synthetic Telco-like rows (offline fallback)...")

    yes_no = ["Yes", "No"]
    internet = ["DSL", "Fiber optic", "No"]
    contracts = ["Month-to-month", "One year", "Two year"]
    payments = [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ]
    addon = ["Yes", "No", "No internet service"]

    fields = [
        "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
        "tenure", "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
        "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn",
    ]

    import string
    rows = []
    for i in range(n):
        tenure = random.randint(0, 72)
        contract = random.choices(contracts, weights=[0.55, 0.21, 0.24])[0]
        monthly = round(random.uniform(18.0, 119.0), 2)
        total = round(monthly * max(tenure, 1) * random.uniform(0.9, 1.05), 2)
        net = random.choices(internet, weights=[0.34, 0.44, 0.22])[0]

        # Churn probability driven by realistic risk factors.
        risk = 0.10
        if contract == "Month-to-month":
            risk += 0.30
        if tenure < 6:
            risk += 0.20
        if net == "Fiber optic":
            risk += 0.10
        churn = "Yes" if random.random() < min(risk, 0.85) else "No"

        cid = f"{random.randint(1000, 9999)}-{\'\'.join(random.choices(string.ascii_uppercase, k=5))}"
        rows.append({
            "customerID": cid,
            "gender": random.choice(["Female", "Male"]),
            "SeniorCitizen": random.choices([0, 1], weights=[0.84, 0.16])[0],
            "Partner": random.choice(yes_no),
            "Dependents": random.choice(yes_no),
            "tenure": tenure,
            "PhoneService": random.choices(yes_no, weights=[0.9, 0.1])[0],
            "MultipleLines": random.choice(["Yes", "No", "No phone service"]),
            "InternetService": net,
            "OnlineSecurity": random.choice(addon),
            "OnlineBackup": random.choice(addon),
            "DeviceProtection": random.choice(addon),
            "TechSupport": random.choice(addon),
            "StreamingTV": random.choice(addon),
            "StreamingMovies": random.choice(addon),
            "Contract": contract,
            "PaperlessBilling": random.choice(yes_no),
            "PaymentMethod": random.choice(payments),
            "MonthlyCharges": monthly,
            "TotalCharges": total,
            "Churn": churn,
        })

    with open(TARGET, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    if TARGET.exists():
        print(f"Telco dataset already present at {TARGET}")
        return

    if _try_kaggle():
        print(f"Saved Telco dataset to {TARGET}")
        return
    if _try_url():
        print(f"Saved Telco dataset to {TARGET}")
        return

    _synthesize()
    print(f"Saved synthetic Telco-like dataset to {TARGET}")
    print(
        "Note: this is synthetic fallback data. For the real dataset, set "
        "KAGGLE_USERNAME/KAGGLE_KEY or TELCO_DATA_URL and re-run."
    )


if __name__ == "__main__":
    sys.exit(main())
