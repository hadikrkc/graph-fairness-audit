# Key findings (auto-generated, Group 14 Mozzarella)

Interpretation: Credit is the primary documented-SEX audit; German is the A92-vs-other grouping sensitivity analysis, not verified female/male membership.
CFA-GCN is a mechanism-inspired approximation, not complete FCLCA. Intervals and persistence indicators are descriptive.
Cross-dataset agreement and portability concern different group definitions; they do not replicate a common sex-fairness effect.
- **RQ1 · German:** recipe-level mean EOdds gains over GCN range -8.2 to 14.2 pp; GCN-SP: 8/27 intervals above zero, 0 intervals below zero, sign reversals in 5/5 splits, beats LogReg on balanced accuracy in 15% of runs, HGB in 34%; GCN-EO: 4/27 intervals above zero, 1 intervals below zero, sign reversals in 5/5 splits, beats LogReg on balanced accuracy in 22% of runs, HGB in 43%; CFA-GCN: 1/27 intervals above zero, 0 intervals below zero, sign reversals in 5/5 splits, beats LogReg on balanced accuracy in 20% of runs, HGB in 43%. η² construction = 0.23 vs method = 0.04; Kendall's W = 0.33.
- **RQ1 · Credit:** recipe-level mean EOdds gains over GCN range -2.6 to 6.7 pp; GCN-SP: 4/27 intervals above zero, 0 intervals below zero, sign reversals in 5/5 splits, beats LogReg on balanced accuracy in 27% of runs, HGB in 22%; GCN-EO: 2/27 intervals above zero, 0 intervals below zero, sign reversals in 5/5 splits, beats LogReg on balanced accuracy in 31% of runs, HGB in 27%; CFA-GCN: 0/27 intervals above zero, 1 intervals below zero, sign reversals in 5/5 splits, beats LogReg on balanced accuracy in 38% of runs, HGB in 33%. η² construction = 0.07 vs method = 0.09; Kendall's W = 0.74.
- **RQ2 · German (legacy A92-vs-other grouping):** excess group homophily (union, k=15) full 0.360, unaware 0.053, legit 0.028; probe AUC inputs 0.731 → 2 hops unaware 0.723, full 0.955; GCN EOdds blind/blind 17.4, pruned topology 10.9, pruned features 10.5, both 10.9 pp.
- **RQ2 · Credit (documented SEX category):** excess group homophily (union, k=15) full 0.403, unaware 0.017, legit 0.016; probe AUC inputs 0.602 → 2 hops unaware 0.586, full 0.692; GCN EOdds blind/blind 9.5, pruned topology 8.2, pruned features 10.0, both 9.8 pp.
- **RQ3 · German:** IER (union, k=15) full 25.3% (same-group 26.9%, cross-group 3.7%); unaware 14.6%. Excess homophily full: union 0.360 → admissible 0.347 vs matched random 0.360.
  - full: retrained ΔEOdds admissible -4.05 ± 3.13 vs matched random -0.92 ± 1.58 pp (all GNNs, all k).
  - unaware: retrained ΔEOdds admissible -1.98 ± 3.05 vs matched random 0.71 ± 1.16 pp (all GNNs, all k).
- **RQ3 · Credit:** IER (union, k=15) full 21.7% (same-group 23.4%, cross-group 0.8%); unaware 8.5%. Excess homophily full: union 0.403 → admissible 0.389 vs matched random 0.403.
  - full: retrained ΔEOdds admissible -0.72 ± 1.38 vs matched random -0.45 ± 0.49 pp (all GNNs, all k).
  - unaware: retrained ΔEOdds admissible 0.10 ± 0.73 vs matched random 0.16 ± 0.52 pp (all GNNs, all k).
- **RQ4 · German selectors (mean over methods/splits):** constraint_audit: bal. acc. 66.4, EOdds 12.7, gain vs GCN -1.68; fairness_only: bal. acc. 68.1, EOdds 12.5, gain vs GCN 1.82; utility_only: bal. acc. 68.0, EOdds 13.0, gain vs GCN 2.93.
  - validation–test Spearman ρ: EOdds -0.00, balanced accuracy 0.35; test-oracle EOdds regret constraint_audit 4.60, fairness_only 4.37, utility_only 4.83; constraint_audit minus fairness_only test EOdds across budgets 0.5 pp: -1.27, 1 pp: +0.30, 2 pp: +0.22, 4 pp: -0.67, 8 pp: -0.28.
- **RQ4 · Credit selectors (mean over methods/splits):** constraint_audit: bal. acc. 68.3, EOdds 8.1, gain vs GCN -0.65; fairness_only: bal. acc. 68.5, EOdds 7.9, gain vs GCN 0.63; utility_only: bal. acc. 69.1, EOdds 7.1, gain vs GCN 1.20.
  - validation–test Spearman ρ: EOdds -0.10, balanced accuracy 0.56; test-oracle EOdds regret constraint_audit 3.24, fairness_only 3.06, utility_only 2.27; constraint_audit minus fairness_only test EOdds across budgets 0.5 pp: +0.67, 1 pp: +0.41, 2 pp: +0.19, 4 pp: +0.80, 8 pp: +0.85.
- **RQ5 · GCN-SP across datasets:** Kendall τ = 0.50, sign agreement 78%, descriptive intervals above zero on both datasets in 3/27 recipes.
- **RQ5 · GCN-EO across datasets:** Kendall τ = 0.29, sign agreement 56%, descriptive intervals above zero on both datasets in 0/27 recipes.
- **RQ5 · CFA-GCN across datasets:** Kendall τ = -0.05, sign agreement 37%, descriptive intervals above zero on both datasets in 0/27 recipes.
- **RQ5 · GCN-SP:** gain vs GCN persists in 1/10 dataset × condition cells; gain vs MLP (≤1 pp balanced-accuracy loss) in 0/10; vs HGB in 0/10.
- **RQ5 · GCN-EO:** gain vs GCN persists in 1/10 dataset × condition cells; gain vs MLP (≤1 pp balanced-accuracy loss) in 0/10; vs HGB in 0/10.
- **RQ5 · CFA-GCN:** gain vs GCN persists in 0/10 dataset × condition cells; gain vs MLP (≤1 pp balanced-accuracy loss) in 0/10; vs HGB in 0/10.
