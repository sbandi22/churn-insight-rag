"""Convert structured Telco churn rows into natural-language summaries.

Embedding raw CSV rows works poorly because numeric/categorical cells lack the
semantic context that a sentence-embedding model needs. Instead we render each
row as a short, fluent paragraph that a retriever and LLM can reason over.
"""
from __future__ import annotations

from typing import Any, Mapping


def _yn(value: Any) -> bool:
    return str(value).strip().lower() in {"yes", "true", "1"}


def _money(value: Any) -> str:
    try:
        return f"${float(value):,.2f}"
    except (TypeError, ValueError):
        return str(value)


def summarize_row(row: Mapping[str, Any]) -> str:
    """Render a single Telco customer record as a natural-language summary."""
    get = lambda k, d="": row.get(k, d)

    gender = str(get("gender", "A customer")).strip()
    senior = "a senior citizen" if str(get("SeniorCitizen")) in {"1", "Yes"} else "not a senior citizen"
    tenure = get("tenure", "an unknown number of")
    contract = str(get("Contract", "unknown")).strip()
    internet = str(get("InternetService", "unknown")).strip()
    payment = str(get("PaymentMethod", "unknown")).strip()
    monthly = _money(get("MonthlyCharges"))
    total = _money(get("TotalCharges"))
    churned = _yn(get("Churn"))

    partner = "has a partner" if _yn(get("Partner")) else "has no partner"
    dependents = "has dependents" if _yn(get("Dependents")) else "has no dependents"

    services = []
    for label, key in [
        ("online security", "OnlineSecurity"),
        ("online backup", "OnlineBackup"),
        ("device protection", "DeviceProtection"),
        ("tech support", "TechSupport"),
        ("streaming TV", "StreamingTV"),
        ("streaming movies", "StreamingMovies"),
    ]:
        if _yn(get(key)):
            services.append(label)
    services_text = ", ".join(services) if services else "no add-on services"

    phone = "has phone service" if _yn(get("PhoneService")) else "has no phone service"
    paperless = "uses paperless billing" if _yn(get("PaperlessBilling")) else "does not use paperless billing"

    churn_text = (
        "This customer has CHURNED (left the company)."
        if churned
        else "This customer has been RETAINED (is still active)."
    )

    cid = get("customerID", "unknown")

    return (
        f"Customer {cid} is a {gender.lower()} customer who is {senior}. "
        f"They have been with the company for {tenure} months on a "
        f"'{contract}' contract. Their internet service is '{internet}' and they "
        f"subscribe to: {services_text}. The customer {phone}, {partner}, and "
        f"{dependents}. They pay {monthly} per month ({total} total to date) via "
        f"'{payment}' and {paperless}. {churn_text}"
    )


def summarize_dataframe(df, max_rows: int | None = None):
    """Yield (summary_text, metadata) tuples for each row of a pandas DataFrame."""
    if max_rows is not None:
        df = df.head(max_rows)
    for idx, row in df.iterrows():
        record = row.to_dict()
        text = summarize_row(record)
        meta = {
            "row_index": int(idx),
            "customer_id": str(record.get("customerID", idx)),
            "churn": str(record.get("Churn", "")),
            "contract": str(record.get("Contract", "")),
        }
        yield text, meta
