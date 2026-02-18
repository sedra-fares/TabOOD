import pandas as pd
from fairlearn.datasets import fetch_acs_income

def load_acs_income():
    # Load 2014
    data_2014 = fetch_acs_income(states=["CA"])
    X14 = pd.DataFrame(data_2014.data, columns=data_2014.feature_names)
    y14 = pd.Series(data_2014.target, name="income")

    # Load 2018
    data_2018 = fetch_acs_income(states=["CA"])
    X18 = pd.DataFrame(data_2018.data, columns=data_2018.feature_names)
    y18 = pd.Series(data_2018.target, name="income")

    majority_mask = y18 == 0

    X_train = pd.concat([X14, X18[majority_mask]])
    y_train = pd.concat([y14, y18[majority_mask]])

    X_test = X18
    y_test = y18

    return X_train, y_train, X_test, y_test


# This is the correct call (no year parameter)
data = fetch_acs_income(states=["CA"], as_frame=True)


# data is a Bunch object → convert to DataFrame + Series
df = data.frame.copy()           # all features + target
X = df.drop(columns=["PINCP"])   # features
y = (df["PINCP"] > 50000).astype(int)  # binary target: 1 = >50k

print("Shape:", X.shape, y.shape)
print("Class balance:", y.value_counts(normalize=True))