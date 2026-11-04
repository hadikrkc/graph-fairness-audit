# Constraint-aware auditing of fairness gains in tabular-to-graph learning

Fair graph neural networks (GNNs) are often evaluated on credit datasets such as German Credit and Credit Default.
These are tables: no network between applicants was ever recorded, and the "graph" is built by the researcher from
attribute similarity. This repository asks whether reported fairness gains are properties of the methods or
by-products of those graph-construction choices.

It contains a single, self-contained notebook that

- builds a multiverse of 27 kNN graph recipes per dataset (3 attribute spaces × 3 graph families × k ∈ {5, 15, 30});
- audits every engineered edge against three declared constraints (screened-space neighbour rank, a training-only
  distance ceiling and a degree cap) and compares the audit with edge-count-matched random controls;
- trains plain and fairness-aware GCNs alongside graph-free references (majority, logistic regression, MLP,
  histogram gradient boosting);
- tests whether fairness gains persist under validation-locked graph selection, inductive scoring of new applicants,
  edge deletion without retraining and transfer to a second dataset.

Target journal: *Knowledge-Based Systems* (Elsevier). The study audits the evaluation setting of
Li, Cheng, Zhang & Zhang (2024), *Contrastive learning for fair graph representations via counterfactual graph
augmentation*, Knowledge-Based Systems 305, 112635, [doi:10.1016/j.knosys.2024.112635](https://doi.org/10.1016/j.knosys.2024.112635).

## Research questions

| RQ | Question |
|---|---|
| RQ1 · Construction dependence | How much do the sign and size of equalised-odds gains of fairness-aware GCNs, relative to an otherwise identical GCN, vary across graph families, neighbourhood sizes and attribute spaces? Do graph models keep any utility advantage over tabular models? |
| RQ2 · Residual group information | Once the group attribute is removed from the inputs, how much group information re-enters through topology versus the remaining node features? |
| RQ3 · Edge-admissibility audit | Do the declared constraints locate edges that exist only because of sensitive or associated information, beyond matched random edge removal? |
| RQ4 · Validation-locked selection | Within a fixed utility budget, does selecting a graph recipe on validation data under the constraints give fairer held-out predictions than selecting on utility or fairness alone? |
| RQ5 · Persistence of claims | Which gains, measured against a plain GCN and against graph-free models, persist under inductive scoring, edge deletion and transfer to the second dataset? |

None of the questions claims causal or counterfactual fairness. The full design is in
[`docs/research_design.md`](docs/research_design.md) and the model/RQ rationale in
[`docs/model_rq_review.md`](docs/model_rq_review.md).

## Data

| Dataset | Source | Use | Group attribute |
|---|---|---|---|
| Statlog (German Credit Data) | [UCI 144](https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data), [doi:10.24432/C5NC77](https://doi.org/10.24432/C5NC77) | all 1,000 records | A92 vs other codes of Attribute 9 (a mixed sex/marital-status field; sensitivity analysis only) |
| Default of Credit Card Clients | [UCI 350](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients), [doi:10.24432/C55S3H](https://doi.org/10.24432/C55S3H) | fixed stratified subset of 5,000 records (seed 20261026) | documented SEX category (primary audit) |

Both datasets are licensed CC BY 4.0. They are **not redistributed** here: the notebook downloads them from UCI and
records SHA-256 checksums. `data/manifests/` holds the checksums, verification reports and the IDs of the Credit subset.
Column documentation is in [`docs/data_dictionary/`](docs/data_dictionary/).

## Repository layout

```
notebooks/p10_graph_fairness_audit.ipynb   the complete experiment (Kaggle-ready, output-free)
results/kaggle_v3/                         outputs of the reported run: figures (PDF), tables (CSV),
                                           raw results, manifest, configuration, environment, key findings
configs/                                   dataset sources and the locked graph-construction grid
scripts/01-03_*.py                         download, verify and subset the data outside the notebook
scripts/04_build_notebook.py               generates the notebook from source
scripts/05_run_notebook.py                 executes the notebook locally and records provenance
scripts/06_verify_run.py                   verifies a completed local run
src/fairgraph/data/                        download and loading helpers used by scripts 01-03
tests/                                     behavioural tests of the selection and control logic
data/manifests/                            checksums, verification report, Credit subset IDs
docs/                                      design, model review, data access, reproducibility, references
```

## How to run

**On Kaggle (as for the reported results).** Import `notebooks/p10_graph_fairness_audit.ipynb`, set
*Accelerator: None* and *Internet: On*, and run all cells. Outputs are written to `/kaggle/working/p10_outputs/`
and zipped as `p10_results.zip`. Without internet access, attach the two UCI CSV exports as a Kaggle dataset; the
notebook finds them by their header row. The full run takes roughly 25–30 minutes on CPU.

**Locally.**

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt -e .
.venv/bin/python scripts/05_run_notebook.py runs/full --jobs 2   # add --quick for a reduced smoke test
.venv/bin/python scripts/06_verify_run.py runs/full
.venv/bin/python -m unittest discover -s tests -v
```

`scripts/05_run_notebook.py` downloads the data unless `--data-dir` points to existing CSVs, and writes an
`execution_record.json` with the source hash, mode and completion state. Quick-mode outputs are smoke tests, not
results. All experimental parameters are fixed in the `CFG` dictionary in the first code cell.

## Results

The reported results are in [`results/kaggle_v3/`](results/kaggle_v3/): 10 figures, 18 tables and the raw
per-run records, listed in [`manifest.csv`](results/kaggle_v3/manifest.csv) and summarised per research question
in [`key_findings.md`](results/kaggle_v3/key_findings.md). All 828 built-in verification checks passed
([`TableS4_method_verification.csv`](results/kaggle_v3/tables/TableS4_method_verification.csv)).

Intervals are computed over five overlapping person-level splits and are descriptive. Results were produced on
Kaggle with Python 3.13.18, NumPy 2.1.3, pandas 2.3.3, scikit-learn 1.6.1 and PyTorch 2.11.0 (CPU). A re-execution
in a different software environment preserves recipe-level gains but not every individual number; see
[`docs/reproducibility.md`](docs/reproducibility.md).

## Limitations

- Edges are engineered similarity, not social relations. The constraints are declared modelling assumptions, not legal
  or causal rules.
- German Credit has no recorded sex column; its group attribute is a coarse derivation and is used as a sensitivity
  analysis only.
- The FCLCA-style CFA-GCN re-implements the mechanism class of the anchor paper; it does not reproduce its published
  numbers.
- Individual numbers are environment-sensitive (see above); conclusions are drawn at recipe level.

## Citation and licence

Code is released under the [MIT licence](LICENSE). Citation metadata is in [`CITATION.cff`](CITATION.cff); a Zenodo
DOI will be added with the first release.

Authors: Abdulhadi Kirkici and Yash Tiwari. We thank Prof. Dr. Raja Hashim Ali for supervision.
