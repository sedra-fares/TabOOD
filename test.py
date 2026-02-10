from fairlearn.datasets import fetch_acs_income
import pandas as pd

from loaders.nsl_kdd_loader import load_nsl_kdd
from preprocessing.common_preprocessor import get_preprocessor

def load_acs_income_with_ood_split():
    data = fetch_acs_income(states=["CA"], as_frame=True)
    df = data.frame.copy()

    # Target: income > 50k
    y = (df["PINCP"] > 50000).astype(int)
    X = df.drop(columns=["PINCP"])

    # Create age groups for shift (middle-aged train, young + old OOD)
    df["age_group"] = pd.cut(
        df["AGEP"],
        bins=[0, 30, 55, 100],
        labels=["young", "middle", "old"],
        include_lowest=True
    )

    train_mask = df["age_group"] == "middle"
    ood_mask   = df["age_group"].isin(["young", "old"])

    X_train = X[train_mask].reset_index(drop=True)
    y_train = y[train_mask].reset_index(drop=True)

    X_ood   = X[ood_mask].reset_index(drop=True)
    y_ood   = y[ood_mask].reset_index(drop=True)

    print("ACS Train shape:", X_train.shape, y_train.shape)
    print("ACS OOD shape: ", X_ood.shape, y_ood.shape)
    print("OOD class balance:\n", y_ood.value_counts(normalize=True))

    return X_train, y_train, X_ood, y_ood


