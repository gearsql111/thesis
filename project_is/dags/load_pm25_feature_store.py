from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
from airflow.operators.trigger_dagrun import TriggerDagRunOperator


default_args = {
    'owner': 'airflow',
    'retries': 5,
    'retry_delay': timedelta(minutes=5),
}

def load_to_feature_store():
    with open('/path/to/local/transformed_data.json', 'r') as f:
        transformed_data = f.read()
    # Load data to feature store, e.g., BigQuery or GCS
    print("Loading data to feature store...")  # Replace with actual loading code

with DAG('load_pm25_feature_store', 
         default_args=default_args, 
         schedule_interval=None, 
         start_date=datetime(2023, 1, 1), 
         tags=['swu'],
         catchup=False) as dag:
    
    load_data_task = PythonOperator(
        task_id='load_to_feature_store',
        python_callable=load_to_feature_store,
    )
