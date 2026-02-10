from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import VarianceThreshold
import pandas as pd
import numpy as np

def get_preprocessor(X: pd.DataFrame) -> Pipeline:
    num_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

    # Remove zero-variance numerical columns to avoid warnings
    num_transformer = Pipeline([
        ('variance', VarianceThreshold(threshold=1e-8)),  # remove near-constant
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_transformer, num_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='if_binary'), cat_cols)
        ],
        remainder='passthrough'
    )

    full_pipe = Pipeline([
        ('preprocessor', preprocessor)
    ])

    return full_pipe