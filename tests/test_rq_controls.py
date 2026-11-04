"""Behavioural checks for v3 controls; the full notebook is verified separately."""
import ast
import unittest
from pathlib import Path

import nbformat

ROOT = Path(__file__).resolve().parents[1]
NB = nbformat.read(ROOT / 'notebooks/p10_graph_fairness_audit.ipynb', as_version=4)
NS = {}
first = next(c.source for c in NB.cells if c.cell_type == 'code')
imports = [n for n in ast.parse(first).body if isinstance(n, (ast.Import, ast.ImportFrom))]
exec(compile(ast.Module(body=imports, type_ignores=[]), '<notebook imports>', 'exec'), NS)
HELPERS = {'Prop', 'SpMM', 'Encoder', 'info_nce', 'soft_eo', 'counterfactual_views',
           'fit', 'predict', 'controlled_selector_contrasts'}
for cell in NB.cells:
    if cell.cell_type != 'code':
        continue
    definitions = [n for n in ast.parse(cell.source).body
                   if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in HELPERS]
    exec(compile(ast.Module(body=definitions, type_ignores=[]), '<notebook helpers>', 'exec'), NS)
NS['FAIR'] = ['GCN-SP', 'GCN-EO', 'CFA-GCN']
NS['CFG'] = dict(hgb_leaf_grid=[7, 15, 31], hgb_iterations=15)
np, pd = NS['np'], NS['pd']


class RQControls(unittest.TestCase):
    def test_notebook_schema(self):
        nbformat.validate(NB)
        for cell in NB.cells:
            if cell.cell_type == 'code':
                self.assertIsNone(cell.execution_count)
                self.assertEqual(cell.outputs, [])
                compile(cell.source, '<cell>', 'exec')

    def fixture(self, feasible=True):
        rows = []
        # Family restriction and minimax must produce different, knowable choices.
        for space, family, k, ba, gap in [
            ('unaware', 'union', 5, 80, 1),
            ('unaware', 'admissible', 5, 79 if feasible else 70, 2),
            ('unaware', 'admissible', 15, 79 if feasible else 70, 20),
            ('legit', 'admissible', 5, 79 if feasible else 70, 4),
            ('legit', 'admissible', 15, 79 if feasible else 70, 5),
        ]:
            rows.append(dict(dataset='Credit', seed=11, method='GCN-EO', space=space,
                             family=family, k=k, val_bal_acc=ba, val_eodds=gap,
                             bal_acc=70+k/10, eodds=gap+3))
        return pd.DataFrame(rows)

    def test_selector_separates_family_and_minimax_without_test_selection(self):
        df = self.fixture()
        a = NS['controlled_selector_contrasts'](df, 2).iloc[0]
        self.assertEqual(a.fairness_only_recipe, 'unaware/union/k5')
        self.assertEqual(a.admissible_only_recipe, 'unaware/admissible/k5')
        self.assertEqual(a.constraint_audit_recipe, 'legit/admissible/k5')
        changed = df.assign(eodds=np.arange(len(df))*-100., bal_acc=np.arange(len(df))*100.)
        b = NS['controlled_selector_contrasts'](changed, 2).iloc[0]
        for name in ['fairness_only_recipe', 'admissible_only_recipe', 'constraint_audit_recipe']:
            self.assertEqual(a[name], b[name])
        self.assertNotEqual(a.family_only_d_eodds, b.family_only_d_eodds)

    def test_infeasible_selector_abstains_with_no_test_contrast(self):
        r = NS['controlled_selector_contrasts'](self.fixture(False), 2).iloc[0]
        self.assertFalse(r.feasible)
        self.assertEqual(r.admissible_candidates, 0)
        self.assertTrue(np.isnan(r.family_only_d_eodds))
        self.assertTrue(np.isnan(r.minimax_d_eodds))

    def test_hgb_choice_ignores_held_out_test_labels_and_groups(self):
        rng = np.random.default_rng(5)
        x = rng.normal(size=(100, 4)).astype(np.float32)
        y = (x[:, 0] + rng.normal(size=100) > 0).astype(int)
        s = rng.integers(0, 2, 100)
        p = dict(Xb=x, split=(np.arange(60), np.arange(60, 80), np.arange(80, 100)))
        a, model_a = NS['fit']('HGB', dict(y=y, s=s), p, seed=11)
        y2, s2 = y.copy(), s.copy()
        y2[80:], s2[80:] = 1-y2[80:], 1-s2[80:]
        b, model_b = NS['fit']('HGB', dict(y=y2, s=s2), p, seed=11)
        np.testing.assert_array_equal(a, b)
        self.assertEqual(model_a.max_leaf_nodes, model_b.max_leaf_nodes)
        self.assertEqual(model_a.p10_validation_candidates_, model_b.p10_validation_candidates_)
        self.assertTrue(np.isfinite(a).all())


if __name__ == '__main__':
    unittest.main()
