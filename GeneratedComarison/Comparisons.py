"""Compare real vs synthetic training data on a fixed real test set.

Runs Logistic Regression, Decision Tree, and XGBoost. Each model is trained
separately on (a) real training data and (b) synthetic training data, and both
are evaluated on the same held-out real test set. Metrics reported: F1, Accuracy,
Precision, Recall.

Usage examples:
  python GeneratedComarison/Comparisons.py --dataset nsl-kdd --synthetic data/synth_nsl.csv
  python GeneratedComarison/Comparisons.py --dataset acs --synthetic data/synth_acs.csv

Synthetic CSV must match the real schema (same columns, including target column
name). Target column name per dataset is set below.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Callable, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

# Ensure project root on path for loaders/preprocessing
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
	sys.path.insert(0, PROJECT_ROOT)

from loaders.nsl_kdd_loader import load_nsl_kdd
from preprocessing.common_preprocessor import get_preprocessor
from test import load_acs_income_with_ood_split


def load_real_data(dataset: str) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, str]:
	"""Load real train/test splits and return (X_train, y_train, X_test, y_test, target_name)."""
	if dataset == "nsl-kdd":
		X_train, y_train, X_test, y_test = load_nsl_kdd(
			"data/nsl_kdd/KDDTrain+.txt",
			"data/nsl_kdd/KDDTest+.txt",
		)
		target = "label"
	elif dataset == "acs":
		# train: middle age, test: young+old from helper
		X_train, y_train, X_test, y_test = load_acs_income_with_ood_split()
		target = "income"
	else:
		raise ValueError(f"Unknown dataset: {dataset}")
	return X_train, y_train, X_test, y_test, target


def load_synthetic(path: str, target: str) -> Tuple[pd.DataFrame, pd.Series]:
	"""Load synthetic CSV and split into X, y using the target column."""
	df = pd.read_csv(path)
	if target not in df.columns:
		raise ValueError(f"Synthetic data missing target column '{target}'")
	y = df[target]
	X = df.drop(columns=[target])
	return X, y


def build_models() -> Dict[str, Callable[[], object]]:
	return {
		"LR": lambda: LogisticRegression(
			max_iter=1000, class_weight="balanced", random_state=42, solver="lbfgs"
		),
		"DT": lambda: DecisionTreeClassifier(
			class_weight="balanced", max_depth=10, random_state=42
		),
		"XGB": lambda: XGBClassifier(
			scale_pos_weight=None,  # filled per-fit when labels provided
			random_state=42,
			eval_metric="logloss",
			n_estimators=200,
			max_depth=6,
			learning_rate=0.1,
		),
	}


def fit_eval(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series, model_fn: Callable[[], object]) -> Dict[str, float]:
	prep = get_preprocessor(X_train)
	X_train_ready = prep.fit_transform(X_train)
	X_test_ready = prep.transform(X_test)

	model = model_fn()

	# For XGB, set scale_pos_weight if binary labels
	if isinstance(model, XGBClassifier):
		pos = (y_train == 1).sum()
		neg = (y_train == 0).sum()
		if pos > 0:
			model.set_params(scale_pos_weight=neg / pos)

	model.fit(X_train_ready, y_train)
	y_pred = model.predict(X_test_ready)

	return {
		"F1": f1_score(y_test, y_pred),
		"Accuracy": accuracy_score(y_test, y_pred),
		"Precision": precision_score(y_test, y_pred),
		"Recall": recall_score(y_test, y_pred),
	}


def compare(real: Tuple[pd.DataFrame, pd.Series], synth: Tuple[pd.DataFrame, pd.Series], test: Tuple[pd.DataFrame, pd.Series]) -> pd.DataFrame:
	models = build_models()
	X_train_real, y_train_real = real
	X_train_synth, y_train_synth = synth
	X_test, y_test = test

	rows = []
	for label, Xtr, ytr in [("Base", X_train_real, y_train_real), ("Synth", X_train_synth, y_train_synth)]:
		for model_name, fn in models.items():
			metrics = fit_eval(Xtr, ytr, X_test, y_test, fn)
			row = {"Train": label, "Model": model_name, **metrics}
			rows.append(row)

	return pd.DataFrame(rows)


def main():
	parser = argparse.ArgumentParser(description="Compare real vs synthetic training on a fixed real test set.")
	parser.add_argument("--dataset", choices=["nsl-kdd", "acs"], default="nsl-kdd")
	parser.add_argument("--synthetic", required=True, help="Path to synthetic CSV matching the real schema, including target column.")
	args = parser.parse_args()

	X_train_real, y_train_real, X_test_real, y_test_real, target = load_real_data(args.dataset)
	X_train_synth, y_train_synth = load_synthetic(args.synthetic, target)

	df = compare((X_train_real, y_train_real), (X_train_synth, y_train_synth), (X_test_real, y_test_real))

	# Order columns and pretty print
	df = df[["Train", "Model", "F1", "Accuracy", "Precision", "Recall"]]
	print("\n=== Comparison (test set = real) ===")
	print(df.to_markdown(index=False, floatfmt=".4f"))

	# Averages per training type
	avg = df.groupby("Train")["F1", "Accuracy", "Precision", "Recall"].mean().reset_index()
	print("\n=== Averages by Train source ===")
	print(avg.to_markdown(index=False, floatfmt=".4f"))


if __name__ == "__main__":
	main()




### To Run: ###
# 1. Generate synthetic data using your model and save as CSV (must match real schema, including target column name).
# 2. Run this script with the --synthetic path to your CSV and --dataset
#    e.g. python GeneratedComarison/Comparisons.py --dataset nsl-kdd --synthetic data/synth_nsl.csv
# 3. Review the printed comparison tables for insights on how synthetic training compares to real training