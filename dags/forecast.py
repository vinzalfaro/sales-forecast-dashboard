import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.model_selection import temporal_train_test_split
from sktime.forecasting.arima import AutoARIMA
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.trend import TrendForecaster


def predictive_analysis():
    
    warnings.filterwarnings("ignore")

    in_df = pd.read_pickle("/opt/airflow/dags/data/processed/clean_monthly_df.pickle")
    in_df["value"] = "2016-2020 actual"

    y_nm = "N_PIECE"

    in_df.index = in_df["year_month"].apply(lambda x: pd.Period(x, freq="M"))

    forecasters_dict = {
        "ARIMA": AutoARIMA(),
        "Linear Regression": TrendForecaster(),
        "Holt's": ExponentialSmoothing(trend="additive", seasonal="additive", sp=12),
    }

    y_pred_df_ls = []
    y_2021_df_ls = []
    for item in in_df["ITEM_DUMMY_ID"].unique():
        sub_df = in_df.loc[in_df["ITEM_DUMMY_ID"] == item, ["ITEM_DUMMY_ID", "value", y_nm]]
        y = sub_df[y_nm]
        y_train, y_test = temporal_train_test_split(y, test_size=0.2)
        fh = ForecastingHorizon(y_test.index, is_relative=False)
        period_2021 = pd.PeriodIndex(
            [pd.Period(f"2021-{i}", freq="M") for i in range(1, 13)]
        )
        fh_2021 = ForecastingHorizon(period_2021, is_relative=False)
        y_pred_df_ls.append(
            pd.DataFrame(
                {"item": item, "model": "actual", "demand_forecast": y, "demand_actual": y}
            )
        )
        for model_name, forecaster in forecasters_dict.items():
            if model_name == "Holt's":
                forecaster.fit(y)
                y_2021_df_ls.append(sub_df)
                y_2021_df_ls.append(
                    pd.DataFrame(
                        {
                            "ITEM_DUMMY_ID": item,
                            "value": "2021 forecast",
                            y_nm: forecaster.predict(fh_2021),
                        }
                    )
                )
            else:
                forecaster.fit(y_train)
            y_pred = forecaster.predict(fh)
            y_pred_df_ls.append(
                pd.DataFrame(
                    {
                        "item": item,
                        "model": model_name,
                        "demand_forecast": y_pred,
                        "demand_actual": y_test,
                    }
                )
            )

    y_pred_df = pd.concat(y_pred_df_ls)
    y_2021_df = pd.concat(y_2021_df_ls)

    y_pred_df["year"] = y_pred_df.index.year
    y_pred_df["month"] = y_pred_df.index.month
    y_pred_df["day"] = 1
    y_pred_df["year_month"] = pd.to_datetime(y_pred_df[["year", "month", "day"]])

    for item in in_df["ITEM_DUMMY_ID"].unique():
        plot_df = y_pred_df.loc[y_pred_df["item"] == item]
        fig, ax = plt.subplots(figsize=(11, 4))
        sns.lineplot(data=plot_df, x="year_month", y="demand_forecast", hue="model", ax=ax)
        ax.set_title(f"Forecasts for {item}")
        plt.savefig(f"/opt/airflow/dags/results/forecasts/plots/{item}.png")

    y_pred_df["demand_forecast"] = y_pred_df["demand_forecast"].round(2)
    test_index = y_pred_df.index[y_pred_df["model"] != "actual"].unique()
    y_pred_df_pivot = (
        y_pred_df.loc[y_pred_df.index.isin(test_index)][
            ["year", "month", "item", "model", "demand_forecast"]
        ]
        .pivot(index=["item", "year", "month"], columns="model", values="demand_forecast")
        .reset_index()
    ).to_csv("/opt/airflow/dags/results/forecasts/test_data_forecasts.csv", index=False)

    metrics_df = y_pred_df.loc[y_pred_df["model"] != "actual"].copy()
    metrics_df["MAD"] = (metrics_df["demand_actual"] - metrics_df["demand_forecast"]).abs()
    metrics_df["MSE"] = (metrics_df["demand_actual"] - metrics_df["demand_forecast"]) ** 2
    metrics_df["MAPE"] = 100 * (metrics_df["MAD"] / metrics_df["demand_actual"].abs())
    metrics_df.loc[metrics_df["demand_actual"] == 0, "MAPE"] = 0
    metrics_df = (
        metrics_df.groupby(["item", "model"], as_index=False, sort=False)[
            ["MAD", "MSE", "MAPE"]
        ]
        .mean()
        .round(2)
    )
    metrics_df.loc[metrics_df["MAPE"] == 0, "MAPE"] = np.nan

    metrics_df.pivot(
        index="item", columns="model", values=["MAD", "MSE", "MAPE"]
    ).reset_index().to_csv("/opt/airflow/dags/results/forecasts/test_data_model_metrics.csv", index=False)

    y_2021_df["year"] = y_2021_df.index.year
    y_2021_df["month"] = y_2021_df.index.month
    y_2021_df["day"] = 1
    y_2021_df["year_month"] = pd.to_datetime(y_2021_df[["year", "month", "day"]])

    y_2021_df = y_2021_df.rename(columns={"ITEM_DUMMY_ID": "item"})

    for item in in_df["ITEM_DUMMY_ID"].unique():
        plot_df = y_2021_df.loc[y_2021_df["item"] == item]
        fig, ax = plt.subplots(figsize=(11, 4))
        sns.lineplot(data=plot_df, x="year_month", y=y_nm, hue="value", ax=ax)
        ax.set_title(f"Holt's 2021 Forecasts for {item}")
        plt.savefig(f"/opt/airflow/dags/results/forecasts/plots_2021/{item}.png")

    y_2021_df["year_month"] = y_2021_df["year_month"].dt.strftime("%Y-%m")
    y_2021_df[y_nm] = y_2021_df[y_nm].round(0).astype(int)

    y_2021_df_pivot = y_2021_df.copy()

    y_2021_df_pivot = y_2021_df_pivot.loc[y_2021_df_pivot["value"] == "2021 forecast"]
    y_2021_df_pivot = y_2021_df_pivot.pivot(
        index="item", columns=["year", "month"], values=y_nm
    ).reset_index()

    y_2021_df_pivot.columns.name = None

    y_2021_df_pivot = y_2021_df_pivot.reset_index(drop=True)

    y_2021_df_pivot.to_csv("/opt/airflow/dags/results/forecasts/Holt_2021_Forecast.csv", index=False)

    s_20_mask = (in_df["ITEM_DUMMY_ID"] == "S-20") & (in_df["year"] == 2019)
    in_df["price_per_item"] = in_df["AMOUNT_PHP"] / in_df["N_PIECE"]
    price_df = (
        in_df.loc[(in_df["year"] == 2020) | s_20_mask]
        .groupby("ITEM_DUMMY_ID", as_index=False)["price_per_item"]
        .last()
    )
    price_df = price_df.rename(columns={"ITEM_DUMMY_ID": "item"})

    y_2021_df = y_2021_df.loc[y_2021_df["value"] == "2021 forecast"]

    n_2021_weekday_holidays = 15
    n_2021_weekdays = 261
    n_2021_working_days = n_2021_weekdays - n_2021_weekday_holidays

    y_2021_annual = y_2021_df.groupby("item", as_index=False)[y_nm].sum()

    y_2021_annual = y_2021_annual.merge(price_df)

    y_2021_annual["AMOUNT_PHP"] = y_2021_annual["N_PIECE"] * y_2021_annual["price_per_item"]

    y_2021_annual["average_daily_demand"] = (
        (y_2021_annual["N_PIECE"] / n_2021_working_days).round().astype(int)
    )

    y_2021_annual["holding_cost"] = y_2021_annual["price_per_item"] * 0.3

    y_2021_annual["order_cost"] = (y_2021_annual["AMOUNT_PHP"] * 0.05).round().astype(int)

    y_2021_annual["EOQ"] = np.sqrt(
        (2 * y_2021_annual["N_PIECE"] * y_2021_annual["order_cost"])
        / y_2021_annual["holding_cost"]
    )
    y_2021_annual["EOQ"] = y_2021_annual["EOQ"].round(2)

    y_2021_annual["ROP"] = (
        y_2021_annual["average_daily_demand"] * 30
        + y_2021_annual["average_daily_demand"] * 1.5
    )

    y_2021_annual.to_csv("/opt/airflow/dags/results/EOQ_ROP_2021_annual.csv", index=False)