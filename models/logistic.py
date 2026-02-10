
import sys
import os

# Add the project root to sys.path so imports work from anywhere
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from loaders.nsl_kdd_loader import load_nsl_kdd
from preprocessing.common_preprocessor import get_preprocessor
from test import load_acs_income_with_ood_split

# Load with OOD split
X_train_acs, y_train_acs, X_ood_acs, y_ood_acs = load_acs_income_with_ood_split()

# Preprocess
prep_acs = get_preprocessor(X_train_acs)
X_train_ready = prep_acs.fit_transform(X_train_acs)
X_ood_ready  = prep_acs.transform(X_ood_acs)

# Train simple logistic regression (balanced because imbalance)
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42,
    solver="lbfgs"
)
model.fit(X_train_ready, y_train_acs)

# Predict on OOD
y_pred_ood = model.predict(X_ood_ready)

# Metrics
print("\n=== ACS Income Baseline (Logistic Regression on OOD) ===")
print(f"Accuracy:  {accuracy_score(y_ood_acs, y_pred_ood):.4f}")
print(f"Precision: {precision_score(y_ood_acs, y_pred_ood):.4f}")
print(f"Recall:    {recall_score(y_ood_acs, y_pred_ood):.4f}")
print(f"F1-score:  {f1_score(y_ood_acs, y_pred_ood):.4f}")


# === NSL-KDD Baseline ===
X_train_nsl, y_train_nsl, X_test_nsl, y_test_nsl = load_nsl_kdd(
    "data/nsl_kdd/KDDTrain+.txt",
    "data/nsl_kdd/KDDTest+.txt"
)

prep_nsl = get_preprocessor(X_train_nsl)
X_train_nsl_ready = prep_nsl.fit_transform(X_train_nsl)
X_test_nsl_ready  = prep_nsl.transform(X_test_nsl)

model_nsl = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)
model_nsl.fit(X_train_nsl_ready, y_train_nsl)

y_pred_nsl = model_nsl.predict(X_test_nsl_ready)

print("\n=== NSL-KDD Baseline (Logistic Regression on OOD Test) ===")
print(f"Accuracy:  {accuracy_score(y_test_nsl, y_pred_nsl):.4f}")
print(f"Precision: {precision_score(y_test_nsl, y_pred_nsl):.4f}")
print(f"Recall:    {recall_score(y_test_nsl, y_pred_nsl):.4f}")
print(f"F1-score:  {f1_score(y_test_nsl, y_pred_nsl):.4f}")


