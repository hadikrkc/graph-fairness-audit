"""Execute the Kaggle notebook locally and save the executed copy.

Usage: python scripts/05_run_notebook.py <workdir> [--quick]
Outputs (p10_outputs/, p10_results.zip, executed notebook) are written to <workdir>.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from jupyter_client import KernelManager

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "notebooks" / "p10_graph_fairness_audit.ipynb"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("workdir", type=Path)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--data-dir", type=Path, help="Reuse previously downloaded UCI CSVs")
    args = parser.parse_args()
    if args.jobs < 1:
        parser.error("--jobs must be positive")
    workdir = args.workdir.resolve()
    workdir.mkdir(parents=True, exist_ok=True)
    # Explicitly reset quick mode: a full run cannot inherit an old shell flag.
    os.environ["P10_QUICK"] = "1" if args.quick else "0"
    os.environ["P10_JOBS"] = str(args.jobs)
    for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
        os.environ[name] = "1"
    for name, folder in [("MPLCONFIGDIR", "matplotlib"), ("IPYTHONDIR", "ipython"), ("JUPYTER_RUNTIME_DIR", "jupyter")]:
        path = workdir / ".runtime" / folder
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    if args.data_dir:
        dest = workdir / "p10_data"
        dest.mkdir(exist_ok=True)
        for filename in ["uci_144.csv", "uci_350.csv"]:
            shutil.copy2(args.data_dir.resolve() / filename, dest / filename)
    nb = nbformat.read(SRC, as_version=4)
    t = time.time()
    out = workdir / SRC.name

    class StreamingClient(NotebookClient):
        def process_message(self, msg, cell, cell_index):
            result = super().process_message(msg, cell, cell_index)
            if msg["msg_type"] == "stream" and msg["content"].get("name") == "stdout":
                print(msg["content"]["text"], end="", flush=True)
            return result

    def cell_start(cell, cell_index, **kwargs):
        print(f"[{(time.time() - t) / 60:.1f} min] executing notebook cell {cell_index + 1}", flush=True)

    def checkpoint(**kwargs):
        nbformat.write(nb, out)

    client = StreamingClient(nb, timeout=7200, kernel_name="python3",
        resources={"metadata": {"path": str(workdir)}}, on_cell_execute=cell_start, on_cell_executed=checkpoint)
    # Use this runner's interpreter, regardless of the user's global kernelspec.
    client.km = KernelManager(kernel_name="python3")
    client.km.kernel_spec.argv[0] = sys.executable
    record = dict(source=str(SRC), source_sha256=hashlib.sha256(SRC.read_bytes()).hexdigest(),
                  interpreter=sys.executable, quick=args.quick, jobs=args.jobs, status="running")
    (workdir / "execution_record.json").write_text(json.dumps(record, indent=2))
    try:
        client.execute()
        record["status"] = "completed"
    except BaseException as err:
        record.update(status="failed", error_type=type(err).__name__, error=str(err)[:2000])
        raise
    finally:
        nbformat.write(nb, out)
        record["runtime_minutes"] = (time.time() - t) / 60
        (workdir / "execution_record.json").write_text(json.dumps(record, indent=2))
        print(f"executed notebook -> {out} ({(time.time() - t) / 60:.1f} min)")
