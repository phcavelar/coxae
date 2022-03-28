from typing import Union
import pandas as pd
import numpy as np
import scipy as sp
import scipy.spatial
try:
    from tqdm.notebook import trange
except ImportError:
    def trange(*args,**kwargs):
        return range(*args)

from sklearn.preprocessing import scale


def stack_dicts(X:dict[str,np.ndarray]) -> np.ndarray:
    return np.concatenate([X[k] for k in X], axis=1)

def preprocess_input_to_dict(X:Union[np.ndarray,dict[str,np.ndarray]]) -> dict[str,np.ndarray]:
    return {"all": X} if not isinstance(X, dict) else X

def remove_constant_columns(df):
    columns_to_remove = get_constant_columns(df)
    return df.drop(columns=columns_to_remove)

def get_constant_columns(df):
    columns_to_remove = []
    for idx, column in enumerate(df.columns):
        try:
            if (df[column].std() == 0).any():
                columns_to_remove.append(column)
        except KeyError:
            columns_to_remove.append(column)
    return columns_to_remove

def remove_columns_with_significant_modes(df, max_mode_pct=0.2):
    columns_to_remove = get_columns_with_significant_modes(df, max_mode_pct=max_mode_pct)
    return df.drop(columns=columns_to_remove)

def get_columns_with_significant_modes(df, max_mode_pct=0.2):
    columns_to_remove = []
    for idx, column in enumerate(df.columns):
        try:
            if (df[column].value_counts(True) >= max_mode_pct).any():
                columns_to_remove.append(column)
        except KeyError:
            columns_to_remove.append(column)
    return columns_to_remove

def remove_columns_with_duplicates(df, eps=1e-3, disable_tqdm=True):
    columns_to_remove = get_columns_with_duplicates(df, eps=eps, disable_tqdm=disable_tqdm)
    return df.drop(columns=columns_to_remove)

def get_columns_with_duplicates(df, eps=1e-3, disable_tqdm=True, catch_keyboard_interrupt=True):
    columns = df.columns
    columns_to_remove = set()
    try:
        try:
            dists = sp.spatial.distance.pdist(df.T.values, "cityblock")
            m = len(columns)
            columns_to_remove = {
                j
                for i in range(m)
                for j in range(i+1,m)
                if dists[m * i + j - ((i + 2) * (i + 1)) // 2] <= eps
            }
        except MemoryError:
            for col1 in trange(len(columns), disable=disable_tqdm):
                if col1 not in columns_to_remove:
                    for col2 in range(col1+1,len(columns)):
                        if abs(df[columns[col1]] - df[columns[col2]]).sum() <= eps:
                            columns_to_remove.add(col2)
    except KeyboardInterrupt as e:
        if not catch_keyboard_interrupt:
            raise e
    return [df.columns[c] for c in columns_to_remove]

def select_top_k_most_variant_columns(df, k, select_least=False):
    argorder = df.var().argsort().values
    return df.iloc[:,argorder[-k:]] if not select_least else df.iloc[:,argorder[:k]]

def maui_scale(df:pd.DataFrame) -> pd.DataFrame:
    """Scale and center data from a DataFrame and return it as a DataFrame. Code based on https://github.com/BIMSBbioinfo/maui/blob/7d329c736b681216093fd725b134a68e6e914c8e/maui/utils.py
    """
    df_scaled = scale(df.T)
    return pd.DataFrame(df_scaled, columns=df.index, index=df.columns).T