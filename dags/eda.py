from itertools import product as iter_product
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from statsmodels.tsa.seasonal import seasonal_decompose


def descriptive_analysis():
    in_df = pd.read_pickle("/opt/airflow/dags/data/processed/clean_df.pickle")

    out_dr = Path("/opt/airflow/dags/results/eda/")

    y_nm = "N_PIECE"
    y2_nm = "AMOUNT_PHP"

    in_df["year"] = in_df["DATE"].dt.year
    in_df["month"] = in_df["DATE"].dt.month

    years = range(2016, 2021)
    months = range(1, 13)
    items = in_df["ITEM_DUMMY_ID"].unique()
    year_month_df = pd.DataFrame(
        iter_product(years, months, items), columns=["year", "month", "ITEM_DUMMY_ID"]
    )

    in_df_monthly = in_df.groupby(["year", "month", "ITEM_DUMMY_ID"], as_index=False)[
        [y_nm, y2_nm]
    ].sum()
    in_df_monthly = year_month_df.merge(in_df_monthly, how="left")
    in_df_monthly.loc[in_df_monthly[y_nm].isna(), [y_nm, y2_nm]] = 0
    in_df_monthly["day"] = 1
    in_df_monthly["year_month"] = pd.to_datetime(in_df_monthly[["year", "month", "day"]])

    for item in [f"S-{i}" for i in range(1, 25)]:
        plot_df = in_df_monthly.loc[in_df_monthly["ITEM_DUMMY_ID"] == item]
        fig, ax = plt.subplots(figsize=(11, 4))
        sns.lineplot(
            data=plot_df,
            x="year_month",
            y=y_nm,
            hue="ITEM_DUMMY_ID",
            ax=ax,
        )
        ax.get_legend().remove()
        ax.set_title(f"Historical demand for {item}")
        plt.savefig(f"/opt/airflow/dags/results/eda/lineplot per SKU/{item}.png")

    ax.get_xticklabels()

    for item in [f"S-{i}" for i in range(1, 6)]:
        plot_df = in_df_monthly.loc[in_df_monthly["ITEM_DUMMY_ID"] == item]
        fig, ax = plt.subplots(figsize=(11, 4))
        sns.lineplot(
            data=plot_df,
            x="year_month",
            y=y_nm,
            hue="ITEM_DUMMY_ID",
            ax=ax,
        )
        ax.get_legend().remove()
        ax.set_title(f"Historical demand for {item}")
        ylims = ax.get_ylim()
        ax.vlines(x=18262.0, ymin=ylims[0], ymax=ylims[1], color="red")
        ax.annotate("\nhigh demand volatility \nstarting 2020", (18300, ylims[1]))
        plt.savefig(f"/opt/airflow/dags/results/eda/edited lineplots/{item}_edited.png")

    in_df_monthly["train_test_year"] = (
        in_df_monthly["year"].astype(str).where(in_df_monthly["year"] == 2020, "2016-2019")
    )
    std_per_year = (
        in_df_monthly.groupby(["ITEM_DUMMY_ID", "train_test_year"], as_index=False)[y_nm]
        .std()
        .round(2)
    )
    std_per_year = std_per_year.pivot(
        index="ITEM_DUMMY_ID", columns="train_test_year", values=y_nm
    ).reset_index()
    std_per_year["std_diff"] = std_per_year["2020"] - std_per_year["2016-2019"]
    std_per_year = std_per_year.sort_values("std_diff", ascending=False)
    std_per_year = std_per_year.drop(columns="std_diff")
    std_per_year.to_csv("/opt/airflow/dags/results/eda/train_test_standard_deviation.csv", index=False)

    zero_prop_df = (
        in_df_monthly.groupby("ITEM_DUMMY_ID", as_index=False)[y_nm]
        .apply(lambda x: (x == 0).sum())
        .rename(columns={y_nm: "months_w_zero_demand"})
    )
    zero_prop_df["zero_prop_%"] = (
        100
        * zero_prop_df["months_w_zero_demand"]
        / in_df_monthly[["year", "month"]].drop_duplicates().shape[0]
    ).round(2)
    zero_prop_df = zero_prop_df.sort_values("zero_prop_%", ascending=False)
    zero_prop_df.to_csv("/opt/airflow/dags/results/eda/proportion of zero values per SKU.csv", index=False)

    xlims = ax.get_xlim()

    null_items = ["S-21", "S-22", "S-23", "S-24"]

    fig, ax = plt.subplots(figsize=(11, 4))
    sns.lineplot(
        data=in_df_monthly.loc[in_df_monthly["ITEM_DUMMY_ID"].isin(null_items)],
        x="year_month",
        y=y_nm,
        hue="ITEM_DUMMY_ID",
        ax=ax,
        alpha=0.7,
    )

    ax.set_xlim(xlims)
    ax.set_title("Demand plot for SKUs w/ large proportion of zeroes")
    plt.savefig("/opt/airflow/dags/results/eda/mostly zero demand SKUs.png")

    fig, ax = plt.subplots(figsize=(11, 4))
    sns.lineplot(
        data=in_df_monthly.loc[~in_df_monthly["ITEM_DUMMY_ID"].isin(null_items)],
        x="year_month",
        y=y_nm,
        hue="ITEM_DUMMY_ID",
        ax=ax,
        alpha=0.9,
    )
    ax.get_legend().remove()
    ax.set_xlim(xlims)
    ax.set_title("Demand plot for SKUs w/ mostly non-zero values")
    plt.savefig("/opt/airflow/dags/results/eda/non-zero demand SKUs.png")

    pareto_df = (
        in_df_monthly.groupby("ITEM_DUMMY_ID", as_index=False)["AMOUNT_PHP"]
        .sum()
        .sort_values(by="AMOUNT_PHP", ascending=False)
    )
    pareto_df["%_share"] = pareto_df["AMOUNT_PHP"] / pareto_df["AMOUNT_PHP"].sum()
    pareto_df["%_cumshare"] = (
        100 * pareto_df["AMOUNT_PHP"].cumsum() / pareto_df["AMOUNT_PHP"].sum()
    )
    pareto_df["%_cumshare"] = pareto_df["%_cumshare"].round(1)

    pareto_df.to_csv("/opt/airflow/dags/results/eda/pareto_df.csv", index=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax2 = ax.twinx()
    sns.barplot(pareto_df, x="ITEM_DUMMY_ID", y="AMOUNT_PHP", ax=ax)
    sns.lineplot(pareto_df, x="ITEM_DUMMY_ID", y="%_cumshare", ax=ax2, color="orange")
    sns.pointplot(
        pareto_df, x="ITEM_DUMMY_ID", y="%_cumshare", ax=ax2, color="orange", markersize=4
    )
    xticks = ax.get_xticks()
    for loc in zip(xticks, pareto_df["%_cumshare"]):
        ax2.annotate(str(loc[1]), loc)
    ax2.grid(axis="y")
    ax.set_title("Sales Pareto Chart")
    plt.savefig("/opt/airflow/dags/results/eda/sales_pareto.png")

    for item in [f"S-{i}" for i in range(1, 25)]:
        plot_sr = (
            in_df_monthly.loc[in_df_monthly["ITEM_DUMMY_ID"] == item]
            .set_index("year_month")[y_nm]
            .resample("M")
            .sum()
        )
        result = seasonal_decompose(plot_sr, model="additive").seasonal
        result_df = pd.DataFrame(result).reset_index()
        result_df["year"] = result_df["year_month"].dt.year
        result_df["month"] = result_df["year_month"].dt.month
        fig, ax = plt.subplots(figsize=(11, 3))
        sns.lineplot(data=result_df.loc[result_df["year"] == 2016], x="month", y="seasonal")
        ax.set_title(f"Seasonal component plot for {item}")
        ax.set_xticks(range(1, 13))
        plt.savefig(f"/opt/airflow/dags/results/eda/seasonal component lineplots/{item}.png")

    for item in [f"S-{i}" for i in range(1, 25)]:
        plot_df = in_df_monthly.loc[in_df_monthly["ITEM_DUMMY_ID"] == item]
        fig, ax = plt.subplots(figsize=(11, 4))
        sns.barplot(
            data=plot_df,
            x="month",
            y=y_nm,
            ax=ax,
        )
        ax.set_title(f"Historical demand distribution by month for {item}")
        plt.savefig(f"/opt/airflow/dags/results/eda/barplot by month/{item}.png")

    in_df_monthly.loc[~in_df_monthly["ITEM_DUMMY_ID"].isin(null_items)].to_pickle(
        "/opt/airflow/dags/data/processed/clean_monthly_df.pickle"
    )