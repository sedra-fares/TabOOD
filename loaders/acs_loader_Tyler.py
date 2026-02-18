import pandas as pd
from fairlearn.datasets import fetch_acs_income
from folktables import ACSDataSource, ACSIncome

def load_acs_income():
    # Load 2014
    data_2014 = fetch_acs_income(states=["CA"])
    data_source = ACSDataSource(survey_year='2014', horizon='1-Year', survey='person')
    acs14 = data_source.get_data(states=['CA'], download=True)
    X_train = pd.DataFrame(acs14.data, columns=acs14.feature_names)
    y_train = pd.Series(acs14.target, name="income")

    # Load 2018
    # data_source = ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
    # acs18 = data_source.get_data(states=['CA'], download=True)
    # X18 = pd.DataFrame(acs18.data, columns=acs18.feature_names)
    # y18 = pd.Series(acs18.target, name="income")

    # majority_mask = y18 == 0
    # features, labels, _ = ACSIncome.df_to_numpy(acs)
    # X_train = pd.DataFrame(features, columns=ACSIncome.feature_names)
    # y_train = pd.Series(labels, name="income")
    # X = pd.DataFrame(features, columns=ACSIncome.feature_names)
    # y = pd.Series(labels, name="income")

    # X_train = pd.concat([X14, X18[majority_mask]])
    # y_train = pd.concat([y14, y18[majority_mask]])

    # X_test = X18
    # y_test = y18

    return X_train, y_train



# # This is the correct call (no year parameter)
data = fetch_acs_income(states=["CA"], cache=True, data_home=None, as_frame=True, return_X_y=False)
# data is a Bunch object → convert to DataFrame + Series
df = data.frame.copy()           # all features + target
X = df.drop(columns=["PINCP"])   # features
y = (df["PINCP"] > 50000).astype(int)  # binary target: 1 = >50k

print("Shape:", X.shape, y.shape)
print("Class balance:", y.value_counts(normalize=True))

