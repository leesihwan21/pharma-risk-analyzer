# Model Comparison

Train: 374,281 rows | Test: 96,543 rows (GroupShuffleSplit on primaryid, no patient overlap)

| Model | Accuracy | F1 (risk) | Recall (risk) | Precision (risk) |
|---|---|---|---|---|
| RandomForest | 0.5082 | 0.4202 | 0.4999 | 0.3624 |
| LightGBM | 0.5185 | 0.4092 | 0.4677 | 0.3637 |
| LogisticRegression | 0.5274 | 0.4058 | 0.4525 | 0.3677 |
| XGBoost | 0.5203 | 0.4052 | 0.4583 | 0.3631 |
