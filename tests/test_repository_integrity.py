"""The public artifact package must verify and reject modified evidence."""
import importlib.util
from pathlib import Path
import shutil
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('repository_integrity', ROOT / 'scripts/07_verify_repository.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class RepositoryIntegrity(unittest.TestCase):
    def test_primary_package_and_source_provenance_verify(self):
        report = module.verify(ROOT)
        self.assertEqual(report['status'], 'verified')
        self.assertEqual(report['code_cells'], 33)
        self.assertEqual(report['primary_output_files'], 47)
        self.assertEqual(report['checksummed_assets'], 51)

    def test_modified_result_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            clone = Path(directory) / 'repository'
            shutil.copytree(ROOT, clone, ignore=shutil.ignore_patterns('.git', '__pycache__', '.venv', 'runs'))
            result = clone / 'results/kaggle_v3/raw_results/runs_grid.csv'
            result.write_bytes(result.read_bytes() + b'\n')
            with self.assertRaisesRegex(AssertionError, 'Artifact hash mismatch'):
                module.verify(clone)


if __name__ == '__main__':
    unittest.main()
