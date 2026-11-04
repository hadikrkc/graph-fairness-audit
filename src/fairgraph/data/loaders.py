"""Load raw files into DataFrames with documented column names."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# UCI documentation (german.doc) attribute names.
GERMAN_COLUMNS = [
    "checking_status", "duration_months", "credit_history", "purpose", "credit_amount",
    "savings_status", "employment_since", "installment_rate", "personal_status_sex",
    "other_debtors", "residence_since", "property", "age", "other_installment_plans",
    "housing", "existing_credits", "job", "num_dependents", "telephone", "foreign_worker",
    "class",
]


def load_german(raw_root: Path) -> pd.DataFrame:
    path = raw_root / "german_credit" / "extracted" / "german.data"
    return pd.read_csv(path, sep=r"\s+", header=None, names=GERMAN_COLUMNS)


def load_credit_default(raw_root: Path) -> pd.DataFrame:
    path = raw_root / "credit_default" / "extracted" / "default of credit card clients.xls"
    # Row 0 holds X1..X23 / Y codes; row 1 holds descriptive names. Keep the X/Y codes
    # so column names match the UCI variable table.
    df = pd.read_excel(path, header=0, skiprows=[1])
    return df.rename(columns={"Unnamed: 0": "ID"})
