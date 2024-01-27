import os
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

files = [
    'results/item_id_mapping.csv',
    'data/processed/clean_df.csv',
    'results/forecasts/test_data_forecasts.csv',
    'results/forecasts/test_data_model_metrics.csv',
    'results/EOQ_ROP_2021_annual.csv'
    ]
dfs = [pd.read_csv(file) for file in files]
db_tables = [file.split("/")[-1].replace(".csv", "") for file in files]


for i in range(len(dfs)):
    print(db_tables[i])
    print(dfs[i].head(5))