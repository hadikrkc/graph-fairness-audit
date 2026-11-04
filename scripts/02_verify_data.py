"""Check raw data against UCI documentation and write data/manifests/data_verification.json."""

import json
from pathlib import Path

import yaml

from fairgraph.data.loaders import load_credit_default, load_german

ROOT = Path(__file__).resolve().parents[1]


def summarise(df, spec, target, sensitive):
    n_features = df.shape[1] - 1 - int("ID" in df.columns)
    return {
        "rows": int(df.shape[0]),
        "features": int(n_features),
        "rows_match_uci": int(df.shape[0]) == spec["expected_rows"],
        "features_match_uci": n_features == spec["expected_features"],
        "missing_values": int(df.isna().sum().sum()),
        "duplicate_rows": int(df.drop(columns=["ID"], errors="ignore").duplicated().sum()),
        "target_distribution": {str(k): int(v) for k, v in df[target].value_counts().sort_index().items()},
        "sensitive_distributions": {
            col: {str(k): int(v) for k, v in df[col].value_counts().sort_index().items()}
            for col in sensitive
        },
    }


if __name__ == "__main__":
    config = yaml.safe_load((ROOT / "configs" / "datasets.yaml").read_text())
    raw = ROOT / "data" / "raw"

    german = load_german(raw)
    credit = load_credit_default(raw)
    report = {
        "german_credit": summarise(german, config["german_credit"], "class",
                                   ["personal_status_sex", "foreign_worker"]),
        "credit_default": summarise(credit, config["credit_default"], "Y", ["X2", "X3", "X4"]),
    }
    report["german_credit"]["age_summary"] = german["age"].describe().round(2).to_dict()
    report["credit_default"]["age_summary"] = credit["X5"].describe().round(2).to_dict()

    out = ROOT / "data" / "manifests" / "data_verification.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
