from datetime import datetime
from airflow import DAG
from airflow.operators.python import PythonOperator

from etl import extract_and_transform, load_to_database
from eda import descriptive_analysis
from forecast import predictive_analysis


local_workflow = DAG(
    "sales_forecast_pipeline",
    schedule_interval="0 6 1 * *",
    start_date=datetime(2024, 1, 1)
)


with local_workflow:
    extract_and_transform_task = PythonOperator(
        task_id="extract_and_transform",
        python_callable=extract_and_transform
    )

    descriptive_analysis_task = PythonOperator(
        task_id="descriptive_analysis",
        python_callable=descriptive_analysis
    )

    predictive_analysis_task = PythonOperator(
        task_id="predictive_analysis",
        python_callable=predictive_analysis
    )

    load_to_database_task = PythonOperator(
        task_id="load_to_database",
        python_callable=load_to_database
    )


    extract_and_transform_task >> descriptive_analysis_task >> predictive_analysis_task >> load_to_database_task