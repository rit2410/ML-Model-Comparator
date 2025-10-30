from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import hashlib


def set_plot_theme():
    sns.set_theme(context="notebook", style="whitegrid")


def dataframe_hash(df: pd.DataFrame) -> str:
# Useful if you later cache based on data content

    h = hashlib.sha256()
    # Stable hash using values and columns
    h.update(pd.util.hash_pandas_object(df, index=True).values)
    h.update("||".join(df.columns).encode())
    return h.hexdigest()
