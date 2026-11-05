# Model–RQ alignment and working revision v3

Review date: 4 November 2026. Scope: local source, existing exploratory results and targeted primary-source literature checks. This is a design review, not a systematic novelty clearance or a guarantee of journal acceptance.

## Assessment

The existing model families are appropriate for a controlled audit of **construction-dependent fairness gains**. The contribution is an explicit edge-audit procedure and its controlled evaluation, rather than a new predictive architecture. More GNN architectures alone would not establish novelty. A negative selection result is an informative RQ4 answer.

| RQ | Required comparison | Existing coverage | v3 change / remaining boundary |
|---|---|---|---|
| RQ1: construction dependence | Same fair method vs matched unregularized backbone on every graph; graph-free utility references | 27 recipes, paired splits, SP/EO/CFA mechanisms, Majority/LogReg/MLP | Add validation-selected histogram gradient boosting (HGB), report the share beating it. Fixed GNN hyperparameters isolate construction; they do not establish a fully tuned SOTA ranking. |
| RQ2: recoverable group information | Predictive inputs vs propagated inputs; crossed removal from features and distance | Training-only field screening, linear probes, 2×2 ablation | Add an attribute-free structural probe (degree, mean neighbour degree, isolation), separately reported. Propagated features do not isolate topology alone, and these structural descriptors cannot bound all topology information. |
| RQ3: edge audit | Admissible vs random removal with identical edge budget and identical fitted model/training seed | Retraining controls; fixed-model interventions | Use 3 random retraining graphs per configuration; give every single filter its own random budget; report paired audit-minus-random effects, averaging draws and k within split. Degree sequence/connectivity remain unmatched. |
| RQ4: graph selection | Same utility floor/objective with and without admissible-family restriction; then change only the objective | Three original validation-only selectors, budget sensitivity, regret | Add a strict supplementary ablation: fairness-only → admissible-only with same fairness objective → admissible minimax objective. Infeasible splits abstain and are counted; conditional comparisons cannot claim population-wide feasibility. Original relaxed selectors remain separately reported. |
| RQ5: persistence | Matched GCN, meaningful graph-free references, inductive/stress/portability evaluation | GCN/MLP, directed query inference, shared deletion masks, source-validation recipe portability | Add HGB utility/fairness comparison to persistence matrix. Different group definitions prevent a claim of replicated sex-fairness effects across datasets. |

## Why these models

- **Majority** exposes trivial low-disparity predictors and class imbalance.
- **LogReg** is an interpretable class-balanced linear tabular reference.
- **MLP** removes message passing while retaining the neural encoder architecture.
- **HGB** adds a nonlinear tree-based reference using the same training-only encoded input. The declared candidates have 7, 15 or 31 leaves; learning rate 0.1, L2=1, 200 iterations, class-balanced loss and no internal early stopping. Select maximum validation balanced accuracy, then fewer leaves. Fit candidates on training records only and never refit using test records. Quick mode uses 60 iterations. This is a limited search, not an assertion that HGB is the globally best tabular learner.
- **GCN-style backbone** is the matched reference for the three fairness mechanisms. It uses row-mean D⁻¹(A+I) propagation, not the original symmetric-normalized Kipf–Welling GCN. All graph methods share it, so the paired construction experiment is controlled; direct published-GCN performance comparisons are inappropriate.
- **GCN-SP / GCN-EO** expose objective dependence: demographic-parity and conditional error-gap penalties are different interventions. Single fixed penalty weights support a controlled comparison, not an optimal utility–fairness frontier.
- **CFA-GCN** supplies a mechanism-inspired contrastive/adversarial comparator. It omits OPT and differs in normalization, adversary and optimization from FCLCA. It must not be labelled a reproduction of FCLCA or used to conclude that published FCLCA fails.

## Defensible novelty statement

> We audit how engineered graph construction changes paired fairness gains, and test a declared rank/distance/degree edge-admissibility procedure against repeated edge-count-matched controls. We separate admissibility restriction from selection objective, and report whether the gains survive graph-free references, inductive inference and structural stress.

The combination is a candidate contribution supported by the reviewed literature, not proof of priority. Varying k, mutual kNN, adding an EO penalty, using a multiverse, or showing that an MLP can beat a GNN are not new individually. Neither generic task-aware graph selection (AutoGrable, 2026), attribute/topology separation (FairGID, 2026), nor fairness-oriented reconstruction of spurious similarity edges (AGFair, 2026) can be claimed as our invention. See `novelty_update_2026-11-04.md` for the new comparisons and the limits of that evidence. Merely combining existing components does not establish a strong journal contribution: the audit must yield a reproducible, useful diagnosis or a substantive limitation supported by the controlled results. The new controls improve evidence; they are not themselves claimed as new ML algorithms.

The most distinctive empirical test is **paired audit-minus-random change at the same edge budget**, with both utility and error gaps. A lower gap caused by a nearly constant predictor is not a useful gain. A negative audit/selection result must remain negative rather than trigger test-informed tuning. Parameters were selected after earlier exploration, so this is not confirmatory preregistration.

Passing the S4 graph invariants only validates implementation against the declared rules. Counting violations of those same rules is not independent evidence that the rules identify ethically invalid or unfair edges. Downstream paired effects and useful diagnostic findings must supply the substantive evidence; the Credit documented-SEX audit is primary and a German legacy-group-only improvement cannot be presented as replicated sex-fairness improvement.

## Interpretation limits that remain

1. Credit's documented SEX grouping is primary; German A92-versus-other is a separate legacy grouping, not verified sex. The two datasets cannot establish demographic-effect replication.
2. The `full` graph and its space-specific distance cutoff use the audit-group indicator. The screened-rank calculation does not directly use endpoint group values, but screening the associated fields uses training-group labels. The pipeline is therefore not demographic-information-free. Do not claim that the full-space audit never uses group information. `unaware`/`legit` attachment does not need a query's group.
3. CFA transductive unsupervised training uses held-out group categories. Its inductive version restricts this to training references. This access differs from SP/EO training and must be disclosed.
4. Five overlapping person splits give descriptive intervals. Random-draw replication does not create independent person samples; draws are averaged before intervals. No confirmatory grid-wide significance claim is justified.
5. The inductive stress test refits selected recipes with training-only reference graphs; recipe selection remains informed by the transductive validation experiment. It is a protocol sensitivity test, not a fully separate nested inductive model-selection benchmark.
6. A full author-code FCLCA/FairGNN/NIFTY benchmark, normalized-GCN sensitivity, broader penalty tuning, degree-preserving controls and more independent cohorts would broaden claims. They are not prerequisites for answering the narrowly stated current RQs, and none is silently asserted to have been completed.

## Primary sources checked

- Qian et al. (KDD 2024), *Addressing Shortcomings in Fair Graph Learning Datasets*: https://arxiv.org/abs/2403.06017 ; official code https://github.com/XweiQ/Benchmark-GraphFairness . Establishes that simple MLPs can outperform GNNs on existing fairness benchmarks; that finding alone is not our novelty.
- Zafeiropoulos et al. (arXiv, 2026), *Investigating the Effect of k-NN Preprocessing on Developing Graph Neural Networks: A Fairness-Based Perspective*: https://arxiv.org/abs/2609.33416 . Already investigates neighbourhood size with an EO-regularized GCN.
- Li et al. (KBS 2024), *Contrastive learning for fair graph representations via counterfactual graph augmentation*: https://doi.org/10.1016/j.knosys.2024.112635 ; author manuscript record https://digital.library.adelaide.edu.au/items/93f9f12d-b8c3-4410-8ec0-1b868ed07be2 . FCLCA combines augmentation, contrastive/adversarial learning and an optimized training strategy; our CFA comparator is incomplete.
- Grinsztajn et al. (NeurIPS 2022), *Why do tree-based models still outperform deep learning on typical tabular data?*: https://papers.neurips.cc/paper_files/paper/2022/file/0378c7692da36807bdec87ab043cdadc-Paper-Datasets_and_Benchmarks.pdf . Motivates a competitive tree reference; does not establish HGB's rank on our cohorts.
- Scikit-learn HGB API: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html . The implementation uses class weighting and a bounded leaf-count search supported by this API.

## Primary result source and execution evidence

The primary source for manuscript figures, tables and numerical findings is **`results/kaggle_v3/`**, archived as `results/p10_results.zip`. Local runs serve reproducibility comparison only.

The primary package contains 1,080 main graph fits, 720 random-control fits, 80 ablation fits, 210 deployment fits and 2,100 stress evaluations, with 10 PDF figures and 18 CSV tables. All 828 built-in checks passed. The downloaded executed Kaggle notebook `notebooks/executed/p10_graph_fairness_audit.kaggle.ipynb` is now verified: all 33 ordered code cells exactly match the working source, execution counts are 1–33 and no error outputs are present. Its environment log, run counts and displayed key findings match the primary result package. Whole-file/code hashes and evidence are in `provenance/kaggle_v3.json`. Public access to the supplied notebook URL (https://www.kaggle.com/code/hadikrkc/p10-graph-fairness-audit) and its page-disclosed version URL (?scriptVersionId=365706511) was verified on recheck on 5 November 2026: HTTP 200, expected title and matching version ID. Earlier 404 observations are historical. This access check does not independently compare remote code bytes; the downloaded local notebook/source match remains separately established.

Kaggle Credit mean test EOdds is 8.09 pp for constraint-audit vs 7.12 pp for utility-only; balanced accuracy is 68.25% vs 69.07%. Persistence against either MLP or HGB is 0/30 cells; the two cells passing against GCN are German diagnostic-full conditions, with no Credit cells passing. Credit paired retrained audit-minus-random EOdds effects, averaging k and methods within split, are −0.269 ± 0.953 pp (full) and −0.065 ± 0.495 pp (unaware). All intervals are descriptive over overlapping splits.

The Kaggle structural probe shows Credit full-space degree/neighbour-degree/isolation AUC 0.72 ± 0.02 on admissible vs 0.58 ± 0.02 on union. Unaware/screened admissible graphs are near 0.50 for this limited probe. This exposes a limitation of the audit rather than evidence that it removes all structural group information.

## RQ4 interpretation and environment sensitivity

RQ4 is a **negative or unstable result**; the manuscript must not claim that the audit selector is generally better. German selector ordering changes between environments, so even its Kaggle ordering is not evidence of a robust advantage. Record this environment sensitivity as a study limitation alongside descriptive intervals, validation fairness noise and flagged floor relaxation. Do not select an environment or a recipe because it produces a more favourable test result.

The primary results were produced with Python 3.13.18, NumPy 2.1.3, pandas 2.3.3, scikit-learn 1.6.1 and torch 2.11.0+cpu; the observed core pins are in `requirements.txt`. In the comparison with a local re-execution (`reproducibility.md`), 79.6% of main-grid runs differ by less than 0.01 pp EOdds, recipe-level mean-gain correlations are r=0.96–1.00 and sign agreement is 96–100%. These observations support broadly preserved recipe-level patterns for the two tested environments, not bitwise equality or a guarantee across arbitrary versions. Detailed environment and numeric contrasts are confined to `docs/reproducibility.md`.

The models and controls support answering the five unchanged RQs. The evidence supports an empirical audit/limitations contribution, not an assured journal novelty claim or a generally superior selection method.
