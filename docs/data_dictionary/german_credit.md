# Data dictionary: Statlog German Credit (UCI 144)

Source file: `data/raw/german_credit/extracted/german.data` (space-separated, no header).
Column names are assigned in `src/fairgraph/data/loaders.py` following `german.doc`.

| # | Column | Type | Role in this study |
|---|---|---|---|
| 1 | checking_status | categorical (A11–A14) | feature |
| 2 | duration_months | integer | feature |
| 3 | credit_history | categorical (A30–A34) | feature |
| 4 | purpose | categorical (A40–A410) | feature |
| 5 | credit_amount | integer | feature |
| 6 | savings_status | categorical (A61–A65) | feature |
| 7 | employment_since | categorical (A71–A75) | feature, possible proxy |
| 8 | installment_rate | integer | feature |
| 9 | personal_status_sex | categorical (A91–A95) | **sensitive** (sex × marital status, A95 absent) |
| 10 | other_debtors | categorical (A101–A103) | feature |
| 11 | residence_since | integer | feature |
| 12 | property | categorical (A121–A124) | feature |
| 13 | age | integer (19–75) | **sensitive** (binarisation threshold must be prespecified) |
| 14 | other_installment_plans | categorical (A141–A143) | feature |
| 15 | housing | categorical (A151–A153) | feature, possible proxy |
| 16 | existing_credits | integer | feature |
| 17 | job | categorical (A171–A174) | feature, possible proxy |
| 18 | num_dependents | integer | feature, possible proxy |
| 19 | telephone | binary (A191–A192) | feature |
| 20 | foreign_worker | binary (A201 yes, A202 no) | **sensitive** (A202 has n = 37) |
| 21 | class | 1 = good, 2 = bad | target |

The "possible proxy" labels are hypotheses for RQ2. They must be confirmed with an association test
(for example mutual information with the sensitive attribute, computed on training data only).
