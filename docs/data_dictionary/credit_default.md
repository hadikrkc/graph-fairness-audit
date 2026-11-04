# Data dictionary: Default of Credit Card Clients (UCI 350)

Source file: `data/raw/credit_default/extracted/default of credit card clients.xls`
(header row = X/Y codes; the descriptive second header row is skipped by the loader).
Taiwan, April–September 2005. Amounts in NT dollars.

| Column | Description | Type | Role in this study |
|---|---|---|---|
| ID | record identifier | integer | key only, never a feature |
| X1 | LIMIT_BAL, credit limit | integer | feature, possible proxy |
| X2 | SEX (1 male, 2 female) | binary | **sensitive** |
| X3 | EDUCATION (1 graduate school, 2 university, 3 high school, 4 others; 0/5/6 undocumented) | categorical | sensitive candidate / proxy |
| X4 | MARRIAGE (1 married, 2 single, 3 others; 0 undocumented) | categorical | sensitive candidate / proxy |
| X5 | AGE (years) | integer | **sensitive** |
| X6–X11 | PAY_0, PAY_2…PAY_6, repayment status Sep→Apr (−2…8) | ordinal | feature |
| X12–X17 | BILL_AMT1…6, bill statement Sep→Apr | integer | feature |
| X18–X23 | PAY_AMT1…6, amount paid Sep→Apr | integer | feature |
| Y | default payment next month (1 yes) | binary | target |

The UCI table calls the first repayment column PAY_0 rather than PAY_1. Keep the X-codes in code to
avoid confusion.

Subset used in experiments: `data/processed/credit_default_subset_5000.csv`
(IDs in `data/manifests/credit_default_subset_ids.txt`).
