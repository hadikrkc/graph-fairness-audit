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
| RQ1 · Construction dependence | When credit records are converted into kNN similarity graphs, how much do the **sign and magnitude** of the equalized-odds gains of fairness-aware GCNs (relative to an otherwise identical GCN) vary across graph families, neighbourhood sizes and attribute spaces, and do the graph models keep any utility advantage over tabular models? |
| RQ2 · Recoverable audit-group information | With the group-defining field excluded from predictive inputs, how does held-out audit-group predictability vary between **node inputs and propagated features**, and how do group error gaps change under crossed removal of associated fields from predictive inputs, graph distances, or both? |
| RQ3 · Edge-admissibility audit | Can declared admissibility constraints (screened-space neighbour rank, distance ceiling, degree cap) identify edges unsupported by the declared non-sensitive feature-space rules? Do models trained or evaluated on admissible edges change group error gaps differently from **matched random edge removal**? |
| RQ4 · Validation-locked graph selection | Under the method-relative validation utility budget, with audit-selector relaxations explicitly reported, how do the three existing selectors compare in held-out utility and group error gaps? |
| RQ5 · Persistence of fairness claims | Which fairness gains, measured against both a plain GCN and a graph-free MLP, persist under **inference on new applicants (inductive)**, edge deletion without retraining, and portability of a source-selected graph recipe to a **second dataset** with separately fitted models? |

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
notebooks/executed/                        byte-identical downloaded, executed Kaggle notebook
results/kaggle_v3/                         outputs of the reported run: figures (PDF), tables (CSV),
                                           raw results, manifest, configuration, environment, key findings
results/p10_results.zip                    original downloadable Kaggle archive
provenance/                               notebook hashes, run evidence and SHA256SUMS
configs/                                   dataset sources and the locked graph-construction grid
scripts/01-03_*.py                         download, verify and subset the data outside the notebook
scripts/04_build_notebook.py               generates the notebook from source
scripts/05_run_notebook.py                 executes the notebook locally and records provenance
scripts/06_verify_run.py                   verifies a completed local run
scripts/07_verify_repository.py            validates the bundled primary results and repository links
src/fairgraph/data/                        download and loading helpers used by scripts 01-03
tests/                                     behavioural tests of the selection and control logic
data/manifests/                            checksums, verification report, Credit subset IDs
docs/                                      design, model review, data access, reproducibility, references
```

## How to run

**On Kaggle (as for the reported results).** Import `notebooks/p10_graph_fairness_audit.ipynb`, set
*Accelerator: None* and *Internet: On*, and run all cells. Outputs are written to `/kaggle/working/p10_outputs/`
and zipped as `p10_results.zip`. Without internet access, attach the two UCI CSV exports as a Kaggle dataset; the
notebook finds them by their header row. The recorded Kaggle run reports 44.0 minutes of compute
time (44.31 minutes total in Papermill metadata); runtime varies by environment.

**Locally on compatible Linux/CPU with Python 3.13.18.** The `.python-version` file records the
reference interpreter. The pins below target the recorded Kaggle core environment, rather than
a complete environment lock; see the reproducibility notes for unrecorded dependency versions.

```bash
python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt -e .
.venv/bin/python scripts/05_run_notebook.py runs/full --jobs 2   # add --quick for a reduced smoke test
.venv/bin/python scripts/06_verify_run.py runs/full
.venv/bin/python scripts/07_verify_repository.py
.venv/bin/python -m unittest discover -s tests -v
```

`scripts/05_run_notebook.py` downloads the data unless `--data-dir` points to existing CSVs, and writes an
`execution_record.json` with the source hash, mode and completion state. Quick-mode outputs are smoke tests, not
results. All experimental parameters are fixed in the `CFG` dictionary in the first code cell. To inspect
the already bundled result package without training models, run `scripts/07_verify_repository.py`.

## Results

The reported results are in [`results/kaggle_v3/`](results/kaggle_v3/): 10 figures, 18 tables and the raw
per-run records, listed in [`manifest.csv`](results/kaggle_v3/manifest.csv) and summarised per research question
in [`key_findings.md`](results/kaggle_v3/key_findings.md). All 828 built-in verification checks passed
([`TableS4_method_verification.csv`](results/kaggle_v3/tables/TableS4_method_verification.csv)).

Intervals are computed over five overlapping person-level splits and are descriptive. Results were produced on
Kaggle with Python 3.13.18, NumPy 2.1.3, pandas 2.3.3, scikit-learn 1.6.1 and PyTorch 2.11.0+cpu. In the observed local/Kaggle comparison, 79.6% of main-grid EOdds differences are below 0.01 pp,
recipe-level mean gains correlate at r=0.96–1.00, and their signs agree in 96–100% of cases.
Individual numbers are not bitwise identical; these findings concern the two tested environments. See
[`docs/reproducibility.md`](docs/reproducibility.md).

## Executed notebook and source provenance

The [executed Kaggle notebook](notebooks/executed/p10_graph_fairness_audit.kaggle.ipynb) contains
33 sequentially executed code cells and no error outputs. Every ordered code cell exactly
matches the output-free source. Both file hashes, the identical code-cell hash, environment
and completion evidence are recorded in [provenance/kaggle_v3.json](provenance/kaggle_v3.json).
The original [results archive](results/p10_results.zip) and
[artifact checksums](provenance/SHA256SUMS) are included.

The [public Kaggle notebook](https://www.kaggle.com/code/hadikrkc/p10-graph-fairness-audit) and
[version 365706511](https://www.kaggle.com/code/hadikrkc/p10-graph-fairness-audit?scriptVersionId=365706511)
were accessible without login/cookies on 5 November 2026 (HTTP 200, expected title/version ID).
Earlier 404 responses remain in the provenance history. Public HTTP access and the local
notebook/source hash comparison are separate checks; remote code was not downloaded in the recheck.

## Limitations

- Edges are engineered similarity, not social relations. The constraints are declared modelling assumptions, not legal
  or causal rules.
- German Credit has no recorded sex column; its group attribute is a coarse derivation and is used as a sensitivity
  analysis only.
- The shared GCN-style backbone uses row-mean propagation, not the canonical symmetric-normalized GCN.
  CFA-GCN is a mechanism-inspired approximation with OPT and other components differing from FCLCA;
  it is not a complete implementation or reproduction of that method.
- RQ4 is negative or unstable: the audit selector has no established general advantage, and German selector
  ordering changes between environments. Do not describe that ordering as robust selector superiority.
- Individual numbers are environment-sensitive (see above); overlapping-split intervals are descriptive.

## Citation and licence

Code is released under the [MIT licence](LICENSE). Citation metadata is in [`CITATION.cff`](CITATION.cff); a Zenodo
DOI will be added with the first release.

Authors: Abdulhadi Kirkici and Yash Tiwari. We thank Prof. Dr. Raja Hashim Ali for supervision.
