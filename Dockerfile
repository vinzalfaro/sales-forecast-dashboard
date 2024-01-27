FROM apache/airflow:2.8.1
ADD requirements.txt .
RUN pip install apache-airflow==2.8.1 -r requirements.txt