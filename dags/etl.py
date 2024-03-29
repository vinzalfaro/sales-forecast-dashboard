import os
import numpy as np
import pandas as pd
from sqlalchemy import create_engine


def extract(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, nrows=11)
    date_loc = np.where(df.values == "DATE")
    header_loc = (date_loc[0][0] + 1, date_loc[1][0])
    return pd.read_excel(path, header=header_loc[0]).iloc[:, header_loc[1]:]


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
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


def extract_and_transform():
    # Set absolute paths based on your folder structure
    raw_dir = "/opt/airflow/dags/data/raw/"
    processed_dir = "/opt/airflow/dags/data/processed/"
    results_dir = "/opt/airflow/dags/results/"

    raw_files = os.listdir(raw_dir)

    in_files = [extract(os.path.join(raw_dir, x)) for x in raw_files]

    clean_files = [prepare_data(x) for x in in_files]

    clean_df = pd.concat(clean_files)

    clean_df["ITEM DESCRIPTION"].value_counts()

    clean_df.loc[clean_df["ITEM DESCRIPTION"] == "SISTER SILK FLOS DAY USE 8X36", "ITEM DESCRIPTION"] = "SISTERS SILK FLOSS DAY USE 8X36"

    items = clean_df["ITEM DESCRIPTION"].unique()
    ids = [f"S-{i}" for i in range(1, len(items) + 1)]
    item_id_dict = dict(zip(items, ids))

    item_id_mapping = pd.DataFrame(item_id_dict.items(), columns=["ITEM DESCRIPTION", "ITEM_DUMMY_ID"])
    item_id_mapping.to_csv(os.path.join(results_dir, "item_id_mapping.csv"), index=False)

    clean_df["ITEM_DUMMY_ID"] = clean_df["ITEM DESCRIPTION"].replace(item_id_dict)
    clean_df.dtypes
    clean_df["PCS_PER_CASE"] = (clean_df["PCS_PER_CASE"].str.extract(r"(\d+)", expand=False).astype(int))
    clean_df["N_PIECE"] = clean_df["N_PIECE"].mask(clean_df["N_PIECE"].isna(), clean_df["PCS_PER_CASE"] * clean_df["N_CASE"])

    clean_df.to_pickle(os.path.join(processed_dir, "clean_df.pickle"))
    clean_df.to_csv(os.path.join(processed_dir, "clean_df.csv"), index=False)


def load_to_database():

    files = [
        '/opt/airflow/dags/results/item_id_mapping.csv',
        '/opt/airflow/dags/data/processed/clean_df.csv',
        '/opt/airflow/dags/results/forecasts/test_data_forecasts.csv',
        '/opt/airflow/dags/results/forecasts/test_data_model_metrics.csv',
        '/opt/airflow/dags/results/EOQ_ROP_2021_annual.csv'
        ]
    
    dfs = [pd.read_csv(file) for file in files]
    db_tables = [file.split("/")[-1].replace(".csv", "") for file in files]

    connection_uri = "postgresql+psycopg2://airflow:airflow@postgres:5432/airflow"
    db_engine = create_engine(connection_uri)

    for i in range(len(dfs)):
        dfs[i].to_sql(
            name = db_tables[i],
            con = db_engine,
            if_exists = "replace",
            index = False)