from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.google.cloud.operators.bigquery import BigQueryInsertJobOperator
from feast import FeatureStore
from datetime import datetime, timedelta
from google.cloud import storage


import os

output_folder = '/opt/airflow/dags/data_test/transformed/'

default_args = {
    'owner': 'airflow',
    'retries': 5,
    'retry_delay': timedelta(minutes=5),
}


def load_to_bigquery():

    gcs_bucket = 'gear-bucket-gcs' 
    dataset_id = 'is-my-project-428015.PM25_dataset'
    table_id = 'pm25_data_hourly'

    client = storage.Client()
    bucket = client.bucket(gcs_bucket)

    for filename in os.listdir(output_folder):
        if filename.endswith('.csv'):
            file_path = os.path.join(output_folder, filename)
            blob = bucket.blob(f"data/{filename}")
            blob.upload_from_filename(file_path)
            print(f"Uploaded {filename} to GCS")

            job_config = {
                "sourceUris": [f"gs://{gcs_bucket}/data/{filename}"],
                "destinationTable": {
                    "projectId": "is-my-project-428015  ",
                    "datasetId": "PM25_dataset",
                    "tableId": "pm25_data_hourly",
                },
                "sourceFormat": "CSV",
                "writeDisposition": "WRITE_APPEND", 
            }

            load_job = BigQueryInsertJobOperator(
                task_id=f"load_{filename}_to_bigquery",
                configuration={"load": job_config},
                location="US",
            )
            load_job.execute(context=None) 


def register_feature_in_feast():
    fs = FeatureStore(repo_path="path/to/your/feature_repo")
    fs.apply()
    print("Feature registered successfully in Feast")


with DAG('load_pm25_data_to_bigquery',
         default_args=default_args,
         schedule_interval=None,
         start_date=datetime(2024, 11, 3),
         tags=['swu'],
         catchup=False) as dag:

    load_task_to_bigquery = PythonOperator(
        task_id='load_to_bigquery',
        python_callable=load_to_bigquery,
    )

    register_feature_task = PythonOperator(
        task_id='register_feature_in_feast',
        python_callable=register_feature_in_feast,
    )

load_task_to_bigquery >> register_feature_task
