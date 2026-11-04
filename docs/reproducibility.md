# Reproducibility and provenance

## Reported run

| Item | Value |
|---|---|
| Source notebook | `notebooks/p10_graph_fairness_audit.ipynb` |
| Source SHA-256 | `0d46f3a274fc9d1d46f5a31e68ea7b65e811e0e47f61cb27b4d8d7ed07adbffa` |
| Configuration revision | `v3_rq_controls_2026-11-04` (`results/kaggle_v3/config.json`) |
| Platform | Kaggle, CPU, Linux x86-64 |
| Environment | Python 3.13.18, NumPy 2.1.3, pandas 2.3.3, scikit-learn 1.6.1, PyTorch 2.11.0+cpu (`results/kaggle_v3/environment.json`) |
| Date | 4 November 2026 |

The environment record lists the core packages only; SciPy, Matplotlib, joblib and threadpoolctl versions were not
recorded and are left unpinned in `requirements.txt`.

The link between this source file and the Kaggle run rests on the configuration revision, the identical
configuration file and the timing of the run. A comparison of the executed Kaggle notebook's code cells with the
source file has not yet been made; it will be added together with the public Kaggle version link.

## Checks on the reported outputs

- 47 files: 10 PDF figures, 18 CSV tables, 15 raw result files, manifest, configuration, environment record and key
  findings. Every manifest path exists, and the downloaded archive matches the extracted files byte for byte.
- All 828 built-in checks passed (`results/kaggle_v3/tables/TableS4_method_verification.csv`): 90 constraint checks,
  240 filter and control-budget checks, 270 symmetry and self-loop checks, 47 inductive-isolation checks,
  180 retraining edge-budget checks and 1 transductive reproduction check.
- Raw records: 1,080 main graph fits, 720 random-control fits, 70 tabular-model records, 80 ablation fits,
  1,280 fixed-model interventions, 210 deployment fits and 2,100 edge-deletion evaluations. No experiment key is
  duplicated, and all performance and gap metrics are finite and within 0–100.
- Selector decisions at all five utility budgets were re-derived from the stored validation metrics.

## Comparison with a local re-execution

The same notebook and configuration were also executed locally (macOS, ARM64; Python 3.14.8, NumPy 2.5.4,
pandas 3.1.0, scikit-learn 1.9.1, PyTorch 2.14.1). Source-data checksums, screened associated fields and the edge
counts of all graph records are identical in both runs. Numerical results are not bitwise identical:

| Comparison | Agreement |
|---|---|
| Matched main-grid runs with absolute EOdds difference below 0.01 pp | 79.6% (mean absolute difference 0.26 pp, maximum 9.81 pp) |
| Pearson correlation of recipe-level mean gains, per dataset and fair method | 0.96–1.00 |
| Agreement of recipe-level mean-gain signs | 96–100% |
| Same graph recipe selected, per dataset and selector | 60–93% |

Differences are more frequent in Credit (32% of its runs differ by at least 0.01 pp, against 9% in German) but
larger in German (95th percentile 2.6 pp against 1.3 pp), whose test groups are small enough for a few changed
predictions to move the equalised-odds gap by several points. Because selectors pick different recipes in some
splits, the local run has 204 deployment fits and 2,040 edge-deletion evaluations instead of 210 and 2,100. The
German ordering of the RQ4 selectors differs between the two environments, so it is not used as evidence of an
advantage for any selector.

Only the Kaggle v3 outputs in `results/kaggle_v3/` are reported. The local run serves as this comparison only.
