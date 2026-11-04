"""Draw the fixed, stratified 5,000-record subset of UCI credit card default.

Writes the subset CSV to data/processed/ and the selected IDs plus checksums to
data/manifests/ so the sampling frame is versioned independently of the raw file.
"""

import json
from pathlib import Path

import yaml
from sklearn.model_selection import train_test_split

from fairgraph.data.download import sha256_of
from fairgraph.data.loaders import load_credit_default

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    cfg = yaml.safe_load((ROOT / "configs" / "datasets.yaml").read_text())["subset"]
    df = load_credit_default(ROOT / "data" / "raw")

    strata = df[cfg["stratify_on"]].astype(str).agg("_".join, axis=1)
    subset, _ = train_test_split(df, train_size=cfg["n"], stratify=strata, random_state=cfg["seed"])
    subset = subset.sort_values("ID")

    out_csv = ROOT / cfg["output"]
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(out_csv, index=False)

    ids_path = ROOT / "data" / "manifests" / "credit_default_subset_ids.txt"
    ids_path.write_text("\n".join(map(str, subset["ID"])) + "\n")

    sub_strata = subset[cfg["stratify_on"]].astype(str).agg("_".join, axis=1)
    record = {
        "source": "credit_default",
        "n": len(subset),
        "seed": cfg["seed"],
        "stratify_on": cfg["stratify_on"],
        "population_strata_share": strata.value_counts(normalize=True).round(4).sort_index().to_dict(),
        "subset_strata_share": sub_strata.value_counts(normalize=True).round(4).sort_index().to_dict(),
        "subset_csv_sha256": sha256_of(out_csv),
        "subset_ids_sha256": sha256_of(ids_path),
    }
    (ROOT / "data" / "manifests" / "credit_default_subset.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=2))
