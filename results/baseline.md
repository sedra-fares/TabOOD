## Baseline Performance on Out-of-Distribution Test Sets

All models trained on in-distribution data and evaluated on OOD/shifted test sets.  
Metrics computed using scikit-learn defaults.

### ACS Income (age-based subpopulation shift)
- Train: ages 30–55  
- OOD Test: ages <30 and >55  

| Model               | Accuracy | Precision | Recall | F1-score |
|---------------------|----------|-----------|--------|----------|
| Logistic Regression | 0.7981   | 0.6740    | 0.7581 | 0.7136   |
| Decision Tree       | 0.8020   | 0.6918    | 0.7275 | 0.7092   |
| XGBoost             | 0.8266   | 0.7367    | 0.7429 | 0.7398   |

### NSL-KDD (unseen attack types)
- Train: KDDTrain+  
- OOD Test: KDDTest+ (contains novel attacks)

| Model               | Accuracy | Precision | Recall | F1-score |
|---------------------|----------|-----------|--------|----------|
| Logistic Regression | 0.7544   | 0.9168    | 0.6253 | 0.7435   |
| Decision Tree       | 0.7752   | 0.9661    | 0.6271 | 0.7606   |
| XGBoost             | 0.7949   | 0.9686    | 0.6611 | 0.7858   |

**Observation**: XGBoost consistently outperforms the other models on both datasets, especially in terms of accuracy and F1-score. Tree-based methods appear more robust to the distribution shifts present in these datasets.
