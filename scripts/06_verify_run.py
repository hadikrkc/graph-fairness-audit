"""Verify a completed local run and reject stale or smoke-only final evidence.

Usage: .venv/bin/python scripts/06_verify_run.py runs/full
Use --allow-quick only to inspect a smoke test. Numerical signs are not acceptance criteria.
"""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

import nbformat
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def verify_run(workdir, allow_quick=False):
    workdir = Path(workdir).resolve()
    record = json.loads((workdir / 'execution_record.json').read_text())
    assert record['status'] == 'completed', 'Run did not complete'
    assert allow_quick or not record['quick'], 'Quick-mode outputs are not final experimental evidence'
    source = ROOT / 'notebooks/p10_graph_fairness_audit.ipynb'
    assert hashlib.sha256(source.read_bytes()).hexdigest() == record['source_sha256'], 'Run source differs from current notebook'
    executed = nbformat.read(workdir / source.name, 4)
    nbformat.validate(executed)
    original = nbformat.read(source, 4)
    assert [c.source for c in executed.cells] == [c.source for c in original.cells], 'Executed source differs'
    if record.get('postprocess'):
        refresh = record['postprocess']
        backup = workdir / refresh['original_executed_notebook']
        assert hashlib.sha256(backup.read_bytes()).hexdigest() == refresh['original_executed_sha256']
        training_nb = nbformat.read(backup, 4)
        changed = [i for i, (a, b) in enumerate(zip(training_nb.cells, original.cells)) if a.source != b.source]
        assert changed == refresh['changed_cell_indices'] and len(changed) == 1
        assert 'save_fig(fig, "Fig09_RQ5_transductive_to_inductive"' in original.cells[changed[0]].source
        figure_nb = nbformat.read(workdir / refresh['refresh_notebook'], 4)
        assert figure_nb.cells[-1].source == original.cells[changed[0]].source
        assert all(c.execution_count is not None for c in figure_nb.cells)
        assert not any(o.output_type == 'error' for c in figure_nb.cells for o in c.outputs)
    code = [c for c in executed.cells if c.cell_type == 'code']
    assert all(c.execution_count is not None for c in code), 'An execution cell is missing'
    assert not any(o.output_type == 'error' for c in code for o in c.outputs), 'Notebook contains an error'
    out = workdir / 'p10_outputs'
    cfg = json.loads((out / 'config.json').read_text())
    assert cfg['revision'] == 'v3_rq_controls_2026-11-04'
    assert len(cfg['seeds']) == (2 if record['quick'] else 5), 'Split count does not match mode'
    manifest = pd.read_csv(out / 'manifest.csv')
    assert manifest.id.is_unique and len(manifest) == 28
    assert (manifest.kind == 'figure').sum() == 10
    assert (manifest.kind == 'table').sum() == 18
    for row in manifest.itertuples():
        path = out / row.file
        assert path.is_file() and path.stat().st_size > 0, f'Missing output: {row.file}'
        if row.kind == 'table':
            assert len(pd.read_csv(path)) > 0, f'Empty table: {row.file}'
    checks = pd.read_csv(out / 'tables/TableS4_method_verification.csv')
    assert checks.all_passed.eq(True).all(), 'A built-in method check failed'
    expected_grid = 2 * len(cfg['seeds']) * len(cfg['spaces']) * len(cfg['families']) * len(cfg['k_grid']) * 4
    grid = pd.read_csv(out / 'raw_results/runs_grid.csv')
    assert len(grid) == expected_grid and not grid.duplicated(['dataset','seed','space','family','k','method']).any()
    random = pd.read_csv(out / 'raw_results/runs_random_matched.csv')
    assert len(random) == 2 * len(cfg['seeds']) * 2 * len(cfg['k_grid']) * cfg['retrain_random_draws'] * 4
    assert not random.duplicated(['dataset','seed','space','k','method','rep']).any()
    with zipfile.ZipFile(workdir / 'p10_results.zip') as archive:
        assert archive.testzip() is None, 'Corrupt archive'
        files = {f'p10_outputs/{p.relative_to(out).as_posix()}': p for p in out.rglob('*') if p.is_file()}
        assert set(archive.namelist()) == set(files), 'Archive has missing or stale files'
        for name, path in files.items():
            assert archive.read(name) == path.read_bytes(), f'Archive differs from output: {name}'
    return dict(status='verified', quick=record['quick'], code_cells=len(code), grid_runs=len(grid),
                random_retraining_runs=len(random), figures=10, tables=18,
                checks=int(checks.times_checked.sum()), runtime_minutes=record['runtime_minutes'])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('workdir', type=Path)
    parser.add_argument('--allow-quick', action='store_true')
    args = parser.parse_args()
    print(json.dumps(verify_run(args.workdir, args.allow_quick), indent=2))
