# Model Comparison

Train: 374,281 rows | Test: 96,543 rows (GroupShuffleSplit on primaryid, no patient overlap)

| Model | Accuracy | F1 (risk) | Recall (risk) | Precision (risk) |
|---|---|---|---|---|
| RandomForest | 0.4937 | 0.4495 | 0.5797 | 0.367 |
| XGBoost | 0.5199 | 0.4098 | 0.4676 | 0.3648 |
| LightGBM | 0.5212 | 0.4069 | 0.4607 | 0.3644 |
| LogisticRegression | 0.5487 | 0.351 | 0.3423 | 0.3601 |
