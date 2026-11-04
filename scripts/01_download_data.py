"""Download both UCI datasets into data/raw/ and write data/manifests/access_log.json."""

from pathlib import Path

import yaml

from fairgraph.data.download import download_dataset, write_manifest

ROOT = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    config = yaml.safe_load((ROOT / "configs" / "datasets.yaml").read_text())
    records = [download_dataset(name, config[name], ROOT / "data" / "raw") for name in ("german_credit", "credit_default")]
    write_manifest(records, ROOT / "data" / "manifests" / "access_log.json")
    for r in records:
        print(f"{r['dataset']}: HTTP {r['http_status']}, {r['archive_bytes']} bytes, sha256 {r['archive_sha256'][:16]}...")
        for f, meta in r["files"].items():
            print(f"    {f}  {meta['bytes']} bytes")
