# %%
import os
from typing import Tuple

import numpy as np
import pandas as pd


# %%
def locate_header(path: str, nrows: int = 11) -> Tuple[int]:
    """
    Locate the header row in an Excel file.

    Parameters
    ----------
    path : str
        Path to the Excel file.
    nrows : int, optional
        Number of rows to read for header detection (default is 11).

    Returns
    -------
    Tuple[int]
        A tuple containing the row index and column index of the "DATE" header.
    """
    df = pd.read_excel(path, nrows=nrows)
    date_loc = np.where(df.values == "DATE")
    return (date_loc[0][0] + 1, date_loc[1][0])


def load_data(path: str) -> pd.DataFrame:
    """
    Load data from an Excel file, and return pandas dataframe with correct headers.

    Parameters
    ----------
    path : str
        Path to the Excel file.

    Returns
    -------
    pd.DataFrame
        Dataframe containing the relevant data.
    """
    header_loc = locate_header(path)
    return pd.read_excel(path, header=header_loc[0]).iloc[:, header_loc[1] :]


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare the loaded data by selecting relevant columns and cleaning it.

    Parameters
    ----------
    df : pd.DataFrame
        Dataframe containing the raw data.

    Returns
    -------
    pd.DataFrame
        Cleaned and processed dataframe with selected columns.
    """
    item_desc_loc = np.where(df.columns == "ITEM DESCRIPTION")[0][0]
    dept_loc = np.where(df.columns == "DEPARTMENT")[0][0]
    case_pc_amt_locs = (dept_loc - np.arange(1, 4)[::-1]).tolist()
    final_col_locs = [0, item_desc_loc, item_desc_loc + 1] + case_pc_amt_locs
    colnames = [
        "DATE",
        "ITEM DESCRIPTION",
        "PCS_PER_CASE",
        "N_CASE",
        "N_PIECE",
        "AMOUNT_PHP",
    ]
    final_df = df.copy().iloc[:, final_col_locs]
    final_df.columns = colnames
    final_df = final_df.dropna(subset=["DATE", "AMOUNT_PHP"])
    final_df["DATE"] = pd.to_datetime(final_df["DATE"])

    return final_df


# %%
raw_dir = "data/raw/"
raw_files = os.listdir(raw_dir)

# %%
in_files = [load_data(os.path.join(raw_dir, x)) for x in raw_files]

# %%
clean_files = [prepare_data(x) for x in in_files]

# %%
clean_df = pd.concat(clean_files)

# %%
clean_df["ITEM DESCRIPTION"].value_counts()

# %%
clean_df.loc[
    clean_df["ITEM DESCRIPTION"] == "SISTER SILK FLOS DAY USE 8X36", "ITEM DESCRIPTION"
] = "SISTERS SILK FLOSS DAY USE 8X36"
# 24 SKUs are left after this renaming

# %%
items = clean_df["ITEM DESCRIPTION"].unique()
ids = [f"S-{i}" for i in range(1, len(items) + 1)]
item_id_dict = dict(zip(items, ids))

# %%
item_id_mapping = pd.DataFrame(
    item_id_dict.items(), columns=["ITEM DESCRIPTION", "ITEM_DUMMY_ID"]
)
item_id_mapping.to_csv("results/item_id_mapping.csv", index=False)

# %%
clean_df["ITEM_DUMMY_ID"] = clean_df["ITEM DESCRIPTION"].replace(item_id_dict)

# %%
clean_df.dtypes

# %% [markdown]
# ## Data preparation
# * parse number of pieces per case from `PCS_PER_CASE` column
# * derive missing values for `N_PIECE` by multiplying number of pieces per case and `N_CASE`

# %%
clean_df["PCS_PER_CASE"] = (
    clean_df["PCS_PER_CASE"].str.extract(r"(\d+)", expand=False).astype(int)
)
clean_df["N_PIECE"] = clean_df["N_PIECE"].mask(
    clean_df["N_PIECE"].isna(), clean_df["PCS_PER_CASE"] * clean_df["N_CASE"]
)

# %% [markdown]
# ## Data dictionary
# * DATE: timestamp - date of transaction
# * ITEM DESCRIPTION: string - name of SKU
# * PCS_PER_CASE: int - number of pieces per case of an SKU
# * N_CASE: int - number of bought cases of an SKU
# * N_PIECE: int - number of bought pieces of an SKU
# * AMOUNT_PHP: float - amount of transaction in Php
# * ITEM_DUMMY_ID: string - dummy id assigned to each SKU. This will be used in place of SKU name.

# %%
clean_df.to_pickle("data/processed/clean_df.pickle")
clean_df.to_csv("data/processed/clean_df.csv", index=False)