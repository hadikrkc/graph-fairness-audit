# Research design (v3, five research questions)

**Working title:** Constraint-aware auditing of fairness gains in tabular-to-graph learning
**Target journal:** Knowledge-Based Systems (Elsevier)
**Anchor paper:** Li, Cheng, Zhang & Zhang (2024), *Contrastive learning for fair graph representations via counterfactual graph augmentation*, KBS 305:112635
**Data:** Credit Default is the primary documented-SEX audit (fixed stratified n=5,000, seed 20261026). German Credit (n=1,000) is a separate legacy A92-versus-other grouping sensitivity analysis. Both datasets are CC BY 4.0.
**Comparability:** The anchor paper uses age and all 30,000 Credit records; this study uses documented SEX categories and 5,000 records. German G is not verified sex. South German Credit documents the same underlying German cohort. Cross-dataset recipe portability does not replicate a common sex-fairness effect.

**Notebook:** `notebooks/p10_graph_fairness_audit.ipynb` (built by `scripts/04_build_notebook.py`; Kaggle-ready)

**Primary reported results:** `results/kaggle_v3/`. All manuscript numbers, figures, tables and active result summaries use this source. Local runs are comparison-only.

## Contribution and position

KBS publishes AI methods for prediction and decision support, including graph-based knowledge representation. The anchor paper is a KBS fair-GNN paper that evaluates on German and Credit, whose edges are **engineered similarity relations**. We treat graph construction as a knowledge-engineering decision that has to be audited. The contribution has three parts:
1. a **constraint-aware edge-admissibility audit** with three declared, checkable constraints (screened-space neighbour rank in a non-sensitive feature space, a training-calibrated distance ceiling, a degree cap);
2. a **validation-locked selection procedure** built on that audit;
3. a **systematic robustness evaluation** of fair-GNN gains: construction multiverse, matched random controls, inductive deployment and edge deletion without retraining.

**Closest work and how we differ:**

| Work | What it does | Difference |
|---|---|---|
| Qian et al. (2024, KDD ADS) | Shows that fair-graph datasets have weak edges and that an MLP can beat GNNs; proposes new datasets | We audit and constrain existing constructions instead of replacing the data |
| De Oliveira & Berton (2025, ENIAC) | Compares kNN, mutual kNN and sequential kNN for fairness in graph-based semi-supervised learning (including German Credit); links degree disparity to unfairness | We study fair-GNN *gains* rather than base-learner fairness, and add sensitive and legitimacy constraints, validation-locked selection, matched random controls and inductive deployment |
| Zafeiropoulos et al. (2026, arXiv 2609.33416) | Varies *k* for an EO-regularised GCN | We vary family, *k* and attribute space jointly and audit edges |

Varying *k*, using mutual kNN and adding an EO penalty are therefore **not** claimed as new.

## Research questions

**RQ1 · Construction dependence** When credit records are converted into kNN similarity graphs, how much do the **sign and magnitude** of the equalized-odds gains of fairness-aware GCNs (relative to an otherwise identical GCN) vary across graph families, neighbourhood sizes and attribute spaces, and do the graph models keep any utility advantage over tabular models?

**RQ2 · Recoverable audit-group information** With the group-defining field excluded from predictive inputs, how does held-out audit-group predictability vary between **node inputs and propagated features**, and how do group error gaps change under crossed removal of associated fields from predictive inputs, graph distances, or both?

**RQ3 · Edge-admissibility audit** Can declared admissibility constraints (screened-space neighbour rank, distance ceiling, degree cap) identify edges unsupported by the declared non-sensitive feature-space rules? Do models trained or evaluated on admissible edges change group error gaps differently from **matched random edge removal**?

**RQ4 · Validation-locked graph selection** Under the method-relative validation utility budget, with audit-selector relaxations explicitly reported, how do the three existing selectors compare in held-out utility and group error gaps?

**RQ5 · Persistence of fairness claims** Which fairness gains, measured against both a plain GCN and a graph-free MLP, persist under **inference on new applicants (inductive)**, edge deletion without retraining, and portability of a source-selected graph recipe to a **second dataset** with separately fitted models?

Each question can return a meaningful negative result. Credit S and German G have different definitions; probe and edge-filter results are not causal fairness claims.

## Planned results (four per research question)

| RQ | Output | Content | Why it matters |
|---|---|---|---|
| RQ1 | Table 1 | Graph multiverse statistics: edges, degree, isolation, components, group degree ratio, sensitive homophily and its excess over random mixing, for 3 spaces × 3 families × k ∈ {5, 15, 30} | Shows how different "the same dataset" is as a graph |
| RQ1 | Fig. 1 | Heat map of the paired EOdds gain (GCN − fair method on the same graph) for every recipe and fair method | Reveals reversals in the direction of the gain |
| RQ1 | Fig. 2 | Specification curve of gains, with a choice-indicator panel | Shows how general a claimed improvement is |
| RQ1 | Table 2 | (A) construction envelope: min/max gain, negative mean gains and intervals excluding zero, splits with a sign reversal, share of runs beating logistic regression; (B) Kendall's W and η² (method vs construction) | Quantifies how much construction drives the result |
| RQ2 | Table 3 | Training-only association (CV AUC, NMI) of every field with dataset-specific audit group, and how often each field is removed | Transparent associated-field rule |
| RQ2 | Fig. 3 | Excess sensitive homophily vs *k* by space (colour) and family (line style) | Separates group-driven from association-driven homophily |
| RQ2 | Fig. 4 | Probe AUC for dataset-specific audit group at 0, 1 and 2 propagation hops, union vs admissible | Recoverable information from combined features and graph structure |
| RQ2 | Table 4 | Crossed 2 × 2 ablation: associated fields removed from node features and/or graph distances, for all GNNs plus MLP/LogReg | Features vs topology channel |
| RQ3 | Fig. 5 | ECDF of the screened-space neighbour rank for same- vs cross-group edges | Shows which edges fail declared non-sensitive feature-space constraints |
| RQ3 | Table 5 | Admissibility audit card: IER (overall, same-group, cross-group), distance and degree violations, group degree ratio, retained share, excess homophily (union, admissible, matched random) | The reusable audit artefact |
| RQ3 | Fig. 6 | Retrained ΔEOdds: union → admissible vs union → density-matched random | Constraint effect beyond density |
| RQ3 | Table 6 | Fixed-model interventions (legitimacy, distance, degree, all constraints vs matched random): ΔEOdds, ΔSP, Δ balanced accuracy, prediction flips | Inference-time effect of the edges, without retraining |
| RQ4 | Table 7 | Utility-only vs fairness-only vs constraint-audit selection: test balanced accuracy, EOdds, gain vs GCN, floor relaxations, plus graph-free references | Does the audit help *selection*? |
| RQ4 | Fig. 7 | Validation vs test EOdds and balanced accuracy for every deployable recipe, with the within-split Spearman rank correlation | Can validation data rank graph recipes at all? |
| RQ4 | Fig. 8 | Test EOdds and balanced accuracy of the three selectors as the utility budget varies (0.5–8 pp) | Is the selector's behaviour an artefact of the 2 pp choice? |
| RQ4 | Table 8 | Selection stability (distinct and modal recipe across splits), regret against a test oracle with the same budget, floor relaxations, validation–test ρ | Is the recommendation reproducible, and how far from the best possible choice is it? |
| RQ5 | Fig. 9 | Accuracy–fairness plane, transductive → inductive arrows for the constraint-selected recipes | Do reported operating points survive new applicants? |
| RQ5 | Fig. 10 | Paired EOdds gain vs edge-deletion rate without retraining (transductive and inductive) | Fragility to edge noise |
| RQ5 | Table 9 | Cross-dataset agreement of recipe-level gains (Kendall τ, sign agreement, recipes with descriptive intervals above zero on both) and portability of the highest mean source-validation-gain recipe without a utility floor, using independent destination fits | Is a gain a property of the method or of one dataset? |
| RQ5 | Table 10 | Persistence matrix: claims (A) vs GCN and (B) vs MLP at ≤ 1 pp mean balanced-accuracy loss, across 5 conditions × 2 datasets (full/union/k15 transductive; selected transductive; selected inductive; selected inductive with 25% deletion; fixed unaware/admissible/k15 transductive); t-CI plus stratified bootstrap | One-glance verdict for each claim |

Supporting tables: S1 provenance, S2 cohorts and group support (including majority-class accuracy), S3 anchor-style reference results, S4 method verification.

## Step-by-step methodology

- **M1 Acquisition and provenance.** Download UCI 144 and 350 (CC BY 4.0), record SHA-256 and assert the documented shapes.
- **M2 Cohorts, outcomes, groups.**
  - German: all 1,000 records, for a separate benchmark-grouping sensitivity analysis. G=1 means Attribute 9=A92; G=0 means any other code. These are not verified female/male memberships. The whole Attribute 9 is excluded from predictive inputs and non-full graph spaces.
  - Credit (primary documented-SEX audit): fixed stratified 5,000 subset (outcome × sex, seed 20261026); this differs from the anchor paper's age-based Credit groups and full cohort. ID, SEX and target are excluded from features.
  - The favourable outcome is coded 1.
- **M3 Splits and training-only preprocessing.** Five person-level 60/20/20 splits (seeds 11, 23, 37, 47, 59), stratified by outcome × group. Encoders and scalers are fit on training persons only, and Credit repayment-status codes are treated as categorical.
- **M4 Associated-field screening.** Training-only 5-fold CV AUC and NMI of every field for the dataset-specific audit group. The top 3 fields form the *associated* set. This is a statistical association, not a causal proxy.
- **M5 Label-free graph construction.**
  - Attribute spaces: *full* (+ standardised audit-group indicator, diagnostic), *unaware* (group field removed), *legit* (group-defining and associated fields removed; screened, not ethically certified).
  - Families: *union* kNN, *mutual* kNN, *admissible* (constraint-filtered union).
  - k ∈ {5, 15, 30}, Euclidean distance.
  - Outcome labels never enter graph construction.
- **M6 Admissibility constraints.** An edge is admissible iff:
  - (i) its normalised screened-space neighbour rank min(rank_i(j), rank_j(i))/k ≤ κ = 2 in the legit space;
  - (ii) its distance is within the training-only 95th percentile of union edge distances;
  - (iii) both endpoints respect a degree cap of 2k, with edges processed in distance order.
  - Audit/base graph isolation is reported and never repaired. CFA training-view augmentation separately repairs isolated nodes and is not guaranteed to preserve the base graph constraints.
- **M7 Models.** Majority, logistic regression, MLP, HGB (validation-only three-candidate leaf-count search), GCN, GCN-SP (statistical-parity penalty), GCN-EO (soft equalized-odds penalty) and CFA-GCN (mechanism-inspired approximation: group-conditioned views + sampled-node InfoNCE + a linear adversary; OPT omitted; author-code availability not verified in this review). Training uses a class-balanced loss, the same two-layer encoder architecture with separately fitted parameters, and early stopping on the validation objective.
- **M8 Metrics.**
  - Utility: balanced accuracy (primary), accuracy, AUC, F1.
  - Fairness: **EOdds gap** = max(|ΔTPR|, |ΔFPR|) (primary), ΔEO (TPR gap, as in the anchor paper) and ΔSP.
  - Gains are paired per split. Intervals are 95% t-intervals over 5 splits, which are descriptive because the splits overlap.
- **M9 Residual information.** Linear probes for the dataset-specific audit group at 0, 1 and 2 propagation hops, plus a crossed 2 × 2 ablation that removes the associated fields from node features and/or graph distances.
- **M10 Audit interventions.**
  - Retrain on admissible graphs versus three density-matched random graphs per dataset/split/space/k (two in quick mode), with draws averaged within configuration before split intervals.
  - Hold the trained model fixed and apply each constraint alone and all together, versus a separately edge-count-matched random deletion for each filter (3 draws).
- **M11 Validation-locked selection.** Three selectors use validation data only:
  - utility-only;
  - fairness-only within a 2 pp balanced-accuracy floor;
  - constraint-audit: admissible family; ranks spaces by worst validation EOdds over all admissible k, then picks an individually utility-feasible recipe. If none is feasible, it automatically relaxes the floor and reports the validation shortfall.
  - Selector diagnostics: within-split rank agreement between validation and test metrics, sensitivity to the utility budget (0.5–8 pp), stability of the chosen recipe across splits, and regret against a test oracle that has the same budget.
- **M12 Deployment and persistence.**
  - Strict inductive inference: the reference graph contains training persons only; validation/test persons receive messages from references, with no feedback.
  - Random edge deletion (0–50%) without retraining, with shared masks for paired models.
  - Cross-dataset recipe portability: select by highest mean source-validation EOdds gain over matched GCN, without a utility constraint; reuse separately fitted destination runs for that recipe. No weights or edges are transferred and no destination-based recipe reselection occurs. Group definitions differ.
  - Stratified person bootstrap (within outcome × group cells) on the primary split.
- **M13 Verification and synthesis.** Built-in checks (symmetry, constraints, inductive isolation, reproducibility of retraining), auto-generated key findings, manifest and zip.

## Limitations to state in the paper

- **RQ4 and environment sensitivity:** treat selection as a negative or unstable result. German
  selector ordering changes across the observed local/Kaggle environments; the manuscript must
  not claim that the audit selector is generally better or present the German ordering as robust
  superiority. Keep environment comparisons separate from primary Kaggle v3 findings. The
  observed result environment is recorded in `requirements.txt` and the primary run's
  `environment.json`; individual results are not bitwise identical across versions.
- Edges are engineered similarity, not social relations. Perturbations and constraints are declared modelling assumptions, not legal or causal rules.
- Five overlapping splits from two historical cohorts. The intervals are descriptive, there are no confirmatory p-values across the grid, and the bootstrap is conditional on the trained models.
- German G is a legacy mixed-code grouping, not verified female/male membership (Grömping, 2019).
- The associated-field rule (top 3 by AUC) is one prespecified choice. Other thresholds are a sensitivity analysis for future work.
- CFA-GCN is a mechanism-inspired approximation with OPT omitted. Transductive augmentation/adversarial training uses held-out categories; inductive CFA uses training-reference categories only. CFA view repair may violate base admissibility.
- An earlier exploratory run (v1, seeds 0–4) informed the design. v2 declares its configuration at run start and uses validation-only recipe selection. Existing exploratory results informed the design; neither fresh seeds nor fixed configuration makes it an independently preregistered confirmatory analysis.

## Exact retained selection and deployment policy

Each fair method uses its own best eligible validation BA as reference. The audit selector ranks spaces by their maximum validation EOdds across all admissible k, including below-floor k; it automatically relaxes a floor with no eligible admissible candidate, reporting the actual shortfall. This is not a common hard utility floor or a controlled comparison changing only the admissibility restriction. Exact-score ties use the notebook's lexicographic keys.

Table 9 has a different source objective: maximize mean source-validation gain over matched GCN without a utility floor. Destination models, screened field identities, encoders and numeric ceilings are fitted locally. There is no destination recipe reselection or model-weight transfer.

Per-space/per-k distance ceilings use the default linear 95th percentile of unique training-only union edges. The cap applies to the undirected base/reference graph; deployment queries have at most k attachments each, without an aggregate reference incoming-query capacity cap. Base isolation is retained; CFA training-view repair is separate.

Figure 6 and pooled RQ3 summaries average configurations within each split before descriptive t-intervals. Figure 7 averages method-specific correlations within split. Kendall W corrects tied ranks. Full local v3 execution and empirical output checks are now documented in `model_rq_review.md`; peak memory was not measured. Isolated synthetic helper checks alone are not experimental results.

## v3 additions (4 November 2026)

The current working notebook keeps the original five RQs, graph grid and three selectors, and adds controls described in `model_rq_review.md`:

- HGB as a graph-free tabular reference (three declared leaf-count candidates; training-only fits, validation balanced-accuracy selection, no test-based choice). Include HGB in RQ1 utility comparisons, RQ4 reference results and RQ5 persistence.
- Three independently sampled edge-count-matched random graphs per retraining configuration (two in quick mode). Each fixed-model filter has its own matched edge budget. Average repetitions within configuration and k within split before intervals.
- S5: direct paired audit-minus-random effects for EOdds and balanced accuracy. Edge-count matching does not control full degree sequence or connectivity.
- S6: strict selector ablation with the same method-relative floor: fairness-only across all families, fairness-only restricted to admissible graphs, then the original admissible minimax objective. Infeasible splits abstain; report their counts and condition contrasts on feasibility. This supplements the original policies with flagged automatic relaxation.
- S7: a separate topology-only probe using degree, mean neighbour degree and isolation, with training-only scaling and logistic regression. It is a limited structural diagnostic, not an exhaustive topology leakage measure.

The shared GCN-style encoder uses row-mean propagation, not canonical symmetric normalization. CFA-GCN remains a mechanism-inspired approximation, not a reproduction of complete FCLCA. The selected-recipe inductive evaluation refits using training-only references, but recipe selection originated in the transductive validation experiment. Novelty must concern the declared audit and its controlled evidence rather than a claim to invent graph selection or feature/topology separation.
