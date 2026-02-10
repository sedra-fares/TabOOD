import sys
import os

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from loaders.nsl_kdd_loader import load_nsl_kdd
from preprocessing.common_preprocessor import get_preprocessor
from test import load_acs_income_with_ood_split   # or from loaders.acs_loader if you moved it

print("=== Decision Tree Baseline (max_depth=10) ===")

# ACS Income
X_train_acs, y_train_acs, X_ood_acs, y_ood_acs = load_acs_income_with_ood_split()
prep_acs = get_preprocessor(X_train_acs)
X_train_ready = prep_acs.fit_transform(X_train_acs)
X_ood_ready = prep_acs.transform(X_ood_acs)

dt_acs = DecisionTreeClassifier(
    class_weight="balanced",
    max_depth=10,
    random_state=42
)
dt_acs.fit(X_train_ready, y_train_acs)
y_pred_dt_acs = dt_acs.predict(X_ood_ready)

print("ACS Income (OOD):")
print(f"Accuracy:  {accuracy_score(y_ood_acs, y_pred_dt_acs):.4f}")
print(f"Precision: {precision_score(y_ood_acs, y_pred_dt_acs):.4f}")
print(f"Recall:    {recall_score(y_ood_acs, y_pred_dt_acs):.4f}")
print(f"F1-score:  {f1_score(y_ood_acs, y_pred_dt_acs):.4f}")

# NSL-KDD
X_train_nsl, y_train_nsl, X_test_nsl, y_test_nsl = load_nsl_kdd(
    "data/nsl_kdd/KDDTrain+.txt",
    "data/nsl_kdd/KDDTest+.txt"
)
prep_nsl = get_preprocessor(X_train_nsl)
X_train_nsl_ready = prep_nsl.fit_transform(X_train_nsl)
X_test_nsl_ready = prep_nsl.transform(X_test_nsl)

dt_nsl = DecisionTreeClassifier(
    class_weight="balanced",
    max_depth=10,
    random_state=42
)
dt_nsl.fit(X_train_nsl_ready, y_train_nsl)
y_pred_dt_nsl = dt_nsl.predict(X_test_nsl_ready)

print("NSL-KDD (OOD):")
print(f"Accuracy:  {accuracy_score(y_test_nsl, y_pred_dt_nsl):.4f}")
print(f"Precision: {precision_score(y_test_nsl, y_pred_dt_nsl):.4f}")
print(f"Recall:    {recall_score(y_test_nsl, y_pred_dt_nsl):.4f}")
print(f"F1-score:  {f1_score(y_test_nsl, y_pred_dt_nsl):.4f}")