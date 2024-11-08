from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from datetime import datetime, timedelta
import pandas as pd
import os


data_folder = '/opt/airflow/dags/data_test/' 
transformed_folder = data_folder + 'transformed/'

default_args = {
    'owner': 'airflow',
    'retries': 5,
    'retry_delay': timedelta(minutes=5),
}

def feature_engineering():
    if not os.path.exists(transformed_folder):
        os.makedirs(transformed_folder)


    for filename in os.listdir(data_folder):
        if filename.endswith('.csv'):
            df = pd.read_csv(os.path.join(data_folder, filename))
            
            df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
            df['day'] = df['date'].dt.day
            
            
            def categorize_pm25(pm25_color_id):
                if pm25_color_id < 2:
                    return 'Very Low'
                elif 2 <= pm25_color_id < 3:
                    return 'Moderate'
                elif 3 <= pm25_color_id < 4:
                    return 'High'
                else:
                    return 'Very High'
            
            
            transformed_file_path = os.path.join(transformed_folder, 'transformed_' + filename)
            df.to_csv(transformed_file_path, index=False)
            print(f"บันทึกไฟล์ที่แปลงแล้วสำเร็จ: {transformed_file_path}")

with DAG('transform_pm25_data',
         default_args=default_args,
         schedule_interval=None,
         start_date=datetime(2024, 11, 3),
         tags=['swu'],
         catchup=False) as dag:


    transform_data_task = PythonOperator(
        task_id='feature_engineering',
        python_callable=feature_engineering,
    )


    trigger_load_data_task = TriggerDagRunOperator(
        task_id='load_to_bigquery',
        trigger_dag_id='load_pm25_feature_store', 
    )

transform_data_task >> trigger_load_data_task
