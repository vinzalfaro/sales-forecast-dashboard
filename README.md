# Sales and Forecast Dashboard

## Project Overview
An extract-transform-load (ETL) pipeline is built to collect the business data. These data are processed to generate insights and forecasts using exploratory data analysis (EDA) and predictive models. The results of the analyses are stored in a PostgreSQL database which connects to a Power BI dashboard.

![Alt text](images/dashboard.png)

Using Airflow, the entire flow of data from the source up to the Power BI dashboard is automated. The scripts are set to run monthly (every first day of the month at 6 AM). 

![Alt text](images/airflow.png)

## To Run
1. Build the docker image:
```bash
docker-compose build
```

2. Initialize Airflow:
```bash
docker-compose up airflow-init
```

3. Run docker-compose:
```bash
docker-compose up
```

4. Go to `localhost:8080` in your browser, enter the credentials, and look for the sales_forecast_pipeline DAG.