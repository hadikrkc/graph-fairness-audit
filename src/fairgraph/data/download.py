"""Download raw UCI archives and record access evidence."""

from __future__ import annotations

import hashlib
import json
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_extract(archive: Path, dest: Path) -> list[str]:
    """Extract a zip, refusing members that would escape `dest`."""
    dest = dest.resolve()
    names = []
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            target = (dest / member.filename).resolve()
            if not target.is_relative_to(dest):
                raise ValueError(f"Unsafe path in archive: {member.filename}")
            zf.extract(member, dest)
            names.append(member.filename)
    return names


def download_dataset(name: str, spec: dict, raw_root: Path) -> dict:
    """Download one dataset archive into raw_root/<name>/ and return an access record."""
    out_dir = raw_root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    archive = out_dir / Path(spec["download_url"]).name.replace("+", "_")

    req = urllib.request.Request(spec["download_url"], headers={"User-Agent": "fairgraph-research/0.1"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        status = resp.status
        archive.write_bytes(resp.read())

    members = safe_extract(archive, out_dir / "extracted")
    files = {
        m: {"sha256": sha256_of(out_dir / "extracted" / m), "bytes": (out_dir / "extracted" / m).stat().st_size}
        for m in members
        if not m.endswith("/")
    }
    return {
        "dataset": name,
        "title": spec["title"],
        "url": spec["download_url"],
        "landing_page": spec["landing_page"],
        "dataset_doi": spec["dataset_doi"],
        "licence": spec["licence"],
        "http_status": status,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "archive": str(archive),
        "archive_sha256": sha256_of(archive),
        "archive_bytes": archive.stat().st_size,
        "files": files,
    }


def write_manifest(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records, indent=2, ensure_ascii=False) + "\n")
