"""Verify the bundled primary run and repository structure without training models.

Usage: python scripts/07_verify_repository.py
The public Kaggle URL/access status is reported separately from local artifact integrity.
"""
import csv
import hashlib
import json
from pathlib import Path
import re
import tomllib
from urllib.parse import unquote, urlsplit
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    with path.open(newline='') as handle:
        return list(csv.DictReader(handle))


def code_cells(notebook):
    return [c for c in notebook['cells'] if c.get('cell_type') == 'code']


def sources(notebook):
    return [''.join(c['source']) if isinstance(c['source'], list) else c['source']
            for c in code_cells(notebook)]


def verify(root=ROOT):
    root = Path(root)
    provenance = json.loads((root / 'provenance/kaggle_v3.json').read_text())
    source_path = root / provenance['working_notebook']
    executed_path = root / provenance['executed_notebook']
    source, executed = [json.loads(p.read_text()) for p in [source_path, executed_path]]
    assert sha256(source_path) == provenance['working_file_sha256'], 'Source hash mismatch'
    assert sha256(executed_path) == provenance['executed_file_sha256'], 'Executed file hash mismatch'
    assert sources(source) == sources(executed), 'Ordered code cells differ'
    code_hash = hashlib.sha256(json.dumps(sources(source), ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()
    assert code_hash == provenance['working_code_sha256'] == provenance['executed_code_sha256']
    cells = code_cells(executed)
    assert len(cells) == 33 and [c['execution_count'] for c in cells] == list(range(1, 34))
    assert not any(o.get('output_type') == 'error' for c in cells for o in c.get('outputs', []))
    assert all(c.get('execution_count') is None and not c.get('outputs') for c in code_cells(source))

    expected_assets = set()
    for line in (root / 'provenance/SHA256SUMS').read_text().splitlines():
        digest, name = line.split('  ', 1)
        path = root / name
        assert path.resolve().is_relative_to(root.resolve()) and path.is_file(), name
        assert sha256(path) == digest, f'Artifact hash mismatch: {name}'
        assert name not in expected_assets, f'Duplicate checksum: {name}'
        expected_assets.add(name)
    out = root / 'results/kaggle_v3'
    files = {p.relative_to(out).as_posix(): p for p in out.rglob('*') if p.is_file() and p.name != '.DS_Store'}
    assert len(files) == 47
    required_assets = {f'results/kaggle_v3/{name}' for name in files}
    required_assets.update({provenance['working_notebook'], provenance['executed_notebook'],
                            'results/p10_results.zip', 'provenance/kaggle_v3.json'})
    assert expected_assets == required_assets, 'Artifact checksum inventory differs'
    with zipfile.ZipFile(root / 'results/p10_results.zip') as archive:
        assert archive.testzip() is None
        assert set(archive.namelist()) == {f'p10_outputs/{name}' for name in files}
        assert all(archive.read(f'p10_outputs/{name}') == path.read_bytes() for name, path in files.items())
    manifest = rows(out / 'manifest.csv')
    assert len(manifest) == 28 and len({row['id'] for row in manifest}) == 28
    assert sum(row['kind'] == 'figure' for row in manifest) == 10
    assert sum(row['kind'] == 'table' for row in manifest) == 18
    assert all((out / row['file']).is_file() for row in manifest)
    checks = rows(out / 'tables/TableS4_method_verification.csv')
    assert all(row['all_passed'].lower() == 'true' for row in checks)
    check_count = sum(int(row['times_checked']) for row in checks)
    assert check_count == 828
    for name, count in {'runs_grid':1080, 'runs_random_matched':720, 'runs_tabular':70,
                        'runs_ablation':80, 'runs_deployment':210, 'runs_edge_deletion':2100}.items():
        assert len(rows(out / f'raw_results/{name}.csv')) == count, name
    environment = json.loads((out / 'environment.json').read_text())
    assert (root / '.python-version').read_text().strip() == environment['python']
    requirements = (root / 'requirements.txt').read_text()
    for name, key in [('numpy','numpy'), ('pandas','pandas'), ('scikit-learn','sklearn'), ('torch','torch')]:
        assert f'{name}=={environment[key]}' in requirements
    for path in [out / 'key_findings.md', out / 'config.json', out / 'environment.json']:
        assert path.stat().st_size > 0

    # README summaries must preserve the canonical five questions.
    design = (root / 'docs/research_design.md').read_text()
    questions = re.findall(r'^\*\*RQ[1-5] .*$', design, re.M)
    readme_questions = re.findall(r'^\| (RQ[1-5] ·.*?) \| (.*?) \|$', (root / 'README.md').read_text(), re.M)
    assert questions == [f'**{title}** {question}' for title, question in readme_questions]
    assert len(questions) == 5
    broken = []
    for path in root.rglob('*.md'):
        if '.git' in path.parts:
            continue
        for target in re.findall(r'\[[^\]]*\]\(([^)]+)\)', path.read_text()):
            target = target.strip('<>')
            parsed = urlsplit(target)
            if parsed.scheme or not parsed.path:
                continue
            dest = (root if parsed.path.startswith('/') else path.parent) / unquote(parsed.path.lstrip('/'))
            if not dest.exists():
                broken.append(f'{path.relative_to(root)}: {target}')
    assert not broken, 'Broken local Markdown links: ' + '; '.join(broken)
    package = tomllib.loads((root / 'pyproject.toml').read_text())
    citation = (root / 'CITATION.cff').read_text()
    citation_version = re.search(r'^version:\s*[\"\x27]?([^\"\x27\s]+)', citation, re.M).group(1)
    assert package['project']['version'] == citation_version
    assert (root / 'LICENSE').is_file()
    return dict(status='verified', code_cells=len(cells), primary_output_files=len(files),
                checksummed_assets=len(expected_assets), built_in_checks=check_count,
                markdown_links='valid', canonical_rqs='unchanged', version=citation_version,
                kaggle_public_access_verified=provenance.get('public_access_verified', False),
                kaggle_http_status=provenance.get('public_access_check', {}).get('http_status'))


if __name__ == '__main__':
    print(json.dumps(verify(), indent=2))
