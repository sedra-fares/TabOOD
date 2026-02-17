import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

# ---------------------------
# Dataset
# ---------------------------
class TabularDataset(Dataset):
    """
    Retourne:
      x_num: FloatTensor (N_num,)
      x_cat: LongTensor  (N_cat,)
      y:     LongTensor  (,)   (optionnel)
    """
    def __init__(self, x_num, x_cat, y=None):
        self.x_num = torch.as_tensor(x_num, dtype=torch.float32)
        self.x_cat = torch.as_tensor(x_cat, dtype=torch.long)
        self.y = None if y is None else torch.as_tensor(y, dtype=torch.long)

    def __len__(self):
        return self.x_num.shape[0]

    def __getitem__(self, idx):
        if self.y is None:
            return self.x_num[idx], self.x_cat[idx]
        return self.x_num[idx], self.x_cat[idx], self.y[idx]


# ---------------------------
# Préprocessing + loaders
# ---------------------------
def make_tabular_loaders(
    df: pd.DataFrame,
    num_cols: list[str],
    cat_cols: list[str],
    label_col: str | None = None,
    batch_size: int = 256,
    test_size: float | None = 0.2,
    val_size: float = 0.1,
    seed: int = 42,
    num_scaler: StandardScaler | None = None,
    cat_encoder: OrdinalEncoder | None = None,
    dropna: bool = True,
    num_workers: int = 0,
    pin_memory: bool = True,
):
    if dropna:
        df = df.dropna(subset=num_cols + cat_cols + ([label_col] if label_col else []))

    idx = np.arange(len(df))

    # -------- TEST SPLIT --------
    if test_size is None or test_size == 0 or test_size == 0.0:
        idx_train = idx
        idx_test = None
    else:
        idx_train, idx_test = train_test_split(
            idx, test_size=test_size, random_state=seed, shuffle=True
        )

    # -------- VAL SPLIT (from train pool only) --------
    if val_size is not None and val_size > 0:
        idx_train, idx_val = train_test_split(
            idx_train, test_size=val_size, random_state=seed, shuffle=True
        )
    else:
        idx_val = None

    def split_df(i):
        return df.iloc[i].reset_index(drop=True)

    df_train = split_df(idx_train)
    df_val   = split_df(idx_val) if idx_val is not None else None
    df_test  = split_df(idx_test) if idx_test is not None else None

    # -------- Fit preprocessors on TRAIN only --------
    if num_scaler is None:
        num_scaler = StandardScaler()
        num_scaler.fit(df_train[num_cols].to_numpy(dtype=np.float32))

    if cat_encoder is None:
        cat_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        cat_encoder.fit(df_train[cat_cols].astype(str)) if len(cat_cols) > 0 else None

    def transform(dfx: pd.DataFrame):
        x_num = num_scaler.transform(dfx[num_cols].to_numpy(dtype=np.float32))
        if len(cat_cols) > 0:
            x_cat = cat_encoder.transform(dfx[cat_cols].astype(str)).astype(np.int64)
            x_cat = np.clip(x_cat, 0, None)
        else:
            x_cat = np.zeros((len(dfx), 0), dtype=np.int64)

        y = None if label_col is None else dfx[label_col].to_numpy()
        return x_num, x_cat, y

    x_num_tr, x_cat_tr, y_tr = transform(df_train)
    x_num_va, x_cat_va, y_va = transform(df_val) if df_val is not None else (None, None, None)
    x_num_te, x_cat_te, y_te = transform(df_test) if df_test is not None else (None, None, None)

    cat_cardinalities = [len(cats) for cats in cat_encoder.categories_] if len(cat_cols) > 0 else []

    ds_train = TabularDataset(x_num_tr, x_cat_tr, y_tr)
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True,
                          num_workers=num_workers, pin_memory=pin_memory, drop_last=False)

    dl_val = None
    if df_val is not None:
        ds_val = TabularDataset(x_num_va, x_cat_va, y_va)
        dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=pin_memory, drop_last=False)

    dl_test = None
    if df_test is not None:
        ds_test = TabularDataset(x_num_te, x_cat_te, y_te)
        dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False,
                             num_workers=num_workers, pin_memory=pin_memory, drop_last=False)

    loaders = {"train": dl_train, "val": dl_val, "test": dl_test}
    preprocessors = {"num_scaler": num_scaler, "cat_encoder": cat_encoder}
    meta = {"cat_cardinalities": cat_cardinalities, "num_cols": num_cols, "cat_cols": cat_cols}
    return loaders, preprocessors, meta