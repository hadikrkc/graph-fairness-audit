# Data access evidence

Checked and downloaded on 26 October 2026. Machine-readable records are in
`data/manifests/access_log.json` (URLs, HTTP status, timestamps, SHA-256 of every file)
and `data/manifests/data_verification.json` (shape and distribution checks).

## Sources

| Dataset | UCI ID | DOI | Licence | Archive SHA-256 | Rows × features | Matches UCI docs |
|---|---|---|---|---|---|---|
| Statlog (German Credit Data) | 144 | [10.24432/C5NC77](https://doi.org/10.24432/C5NC77) | CC BY 4.0 | `e12d9d5def6845c0…` | 1,000 × 20 | yes |
| Default of Credit Card Clients | 350 | [10.24432/C55S3H](https://doi.org/10.24432/C55S3H) | CC BY 4.0 | `56c885f84457f668…` | 30,000 × 23 | yes |

Both datasets are anonymised, publicly released and licensed under CC BY 4.0, which permits reuse
with attribution. Cite both DOIs and, for credit default, Yeh & Lien (2009),
[10.1016/j.eswa.2007.12.020](https://doi.org/10.1016/j.eswa.2007.12.020).
No human-participant recruitment is involved.

## Fixed sampling frame

The credit default subset is drawn by `scripts/03_make_subset.py` from `configs/datasets.yaml`:
n = 5,000, seed = 20261026, stratified on target `Y` × `X2` (sex).
The selected IDs are in `data/manifests/credit_default_subset_ids.txt`
(SHA-256 `d7873e5f2f1e25d2…`). Stratum shares in the subset match the population to within 0.0001.

| Stratum (Y_sex) | Population | Subset |
|---|---|---|
| 0_1 (no default, male) | 0.3005 | 0.3006 |
| 0_2 (no default, female) | 0.4783 | 0.4782 |
| 1_1 (default, male) | 0.0958 | 0.0958 |
| 1_2 (default, female) | 0.1254 | 0.1254 |

## Data-quality findings that affect the study design

1. **German Credit does not support recovering verified sex labels from Attribute 9.**
   The corrected [UCI South German Credit documentation](https://archive.ics.uci.edu/dataset/573/south+german+credit)
   states that `personal_status_sex` combines sex and marital status and that sex cannot be recovered:
   male singles and female non-singles share a code. Therefore, the legacy A92-versus-other grouping
   used in the current notebook is a benchmark grouping proxy, not verified female/male membership.
   Disparities and predictive probes for this grouping must not be interpreted as verified sex-based
   findings. Credit Default has a separately documented SEX field and is the clearer dataset for the
   sex-based audit. South German Credit corrects documentation for the same underlying German data;
   it is not an independent replication cohort.
2. **Group imbalance in German Credit.** Foreign worker = no (A202) has only 37 records out of 1,000,
   which is too few for stable group metrics without resampling or confidence intervals.
3. **Undocumented codes in credit default.** EDUCATION (`X3`) contains 0, 5 and 6, which the UCI
   documentation does not define (14 + 280 + 51 records). MARRIAGE (`X4`) contains 0 (54 records).
   Treat these as "other/unknown" and state the rule in the protocol before any modelling.
4. **35 duplicate feature rows in credit default** (ignoring ID). These can create zero-distance edges
   in a k-NN graph, so the graph-construction protocol must state how ties are handled.
5. **No missing values** in either dataset.

## Anchor paper

Li, C., Cheng, D., Zhang, G., & Zhang, S. (2024). Contrastive learning for fair graph representations
via counterfactual graph augmentation. *Knowledge-Based Systems*, 305, 112635.
[10.1016/j.knosys.2024.112635](https://doi.org/10.1016/j.knosys.2024.112635)

The publisher-version PDF is available in this project at
`paper/1-s2.0-S0950705124012693-main.pdf`. Attach that file when asking for a paper review.
Local file availability does not establish that a separate conversation successfully received or read
the upload. If a reviewer uses an author-accepted manuscript instead, distinguish that version from
the published copy and verify any relevant differences against the publisher PDF.

The download records above document access on 26 October 2026. They do not establish the current
access status of publisher or repository pages. The manifests are supporting provenance records;
checking a narrative summary alone is not independent verification of their contents or file hashes.
