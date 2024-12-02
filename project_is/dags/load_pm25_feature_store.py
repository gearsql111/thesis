from airflow import DAG
from google.oauth2 import service_account
from airflow.operators.python import PythonOperator
from airflow.providers.google.cloud.operators.bigquery import BigQueryInsertJobOperator
from airflow.providers.google.cloud.hooks.bigquery import BigQueryHook
from airflow.providers.google.cloud.hooks.gcs import GCSHook
from feast import FeatureStore
from datetime import datetime, timedelta
from google.cloud import storage


import os
import logging


os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/opt/airflow/credentials/is-my-project-428015-8661db9b13f2.json"
output_folder = "/opt/airflow/dags/data_test/transformed/"
gcs_bucket = "gear-bucket-gcs"
dataset_id = "PM25_dataset"
table_id = "pm25_data_hourly"
project_id = "is-my-project-428015"


default_args = {
    "owner": "airflow",
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
}

# ฟังก์ชันสำหรับอัปโหลดไฟล์ไปยัง GCS และโหลดข้อมูลไปยัง BigQuery
def upload_files_to_gcs():
    gcs_hook = GCSHook(gcp_conn_id="gcp_conn")
    print("Initialized GCSHook")
    
    client = storage.Client()
    bucket = client.bucket(gcs_bucket)
    print("Connected to GCS bucket:", gcs_bucket)

    for filename in os.listdir(output_folder):
        if filename.endswith(".csv"):
            file_path = os.path.join(output_folder, filename)
            blob = bucket.blob(f"data/{filename}")
            blob.upload_from_filename(file_path)
            print(f"Uploaded {filename} to GCS")

    print("Completed upload_files_to_gcs function")

fs = FeatureStore(repo_path="/opt/airflow/feast")
# ฟังก์ชันสำหรับลงทะเบียนฟีเจอร์ใน Feast
def register_feature_in_feast():
    fs.apply(["pm25_features", "station"])
    print("Feature registered successfully in Feast")
    
def ingest_incremental_data():
    fs.materialize_incremental(end_date=datetime.now())
    print("Data materialized into offline store.")

# กำหนด DAG และ Tasks
with DAG(
    "load_pm25_feature_store",
    default_args=default_args,
    schedule_interval=None,
    start_date=datetime(2024, 11, 3),
    tags=["swu"],
    catchup=False,
) as dag:

    # Task สำหรับอัปโหลดไฟล์ไปยัง GCS
    upload_task_to_gcs = PythonOperator(
        task_id="upload_files_to_gcs",
        python_callable=upload_files_to_gcs,
        execution_timeout=timedelta(minutes=30),
    )

    # Task สำหรับโหลดข้อมูลไปยัง BigQuery โดยใช้ BigQueryInsertJobOperator
    load_tasks_to_bigquery = []
    for filename in os.listdir(output_folder):
        if filename.endswith(".csv"):
            load_task = BigQueryInsertJobOperator(
                task_id=f"load_{filename}_to_bigquery",
                configuration={
                    "load": {
                        "sourceUris": [f"gs://{gcs_bucket}/data/{filename}"],
                        "destinationTable": {
                            "projectId": project_id,
                            "datasetId": dataset_id,
                            "tableId": table_id,
                        },
                        "sourceFormat": "CSV",
                        "writeDisposition": "WRITE_APPEND",
                        "autodetect": True,
                        # "schema": [
                        #     {"name": "station_id", "type": "STRING", "mode": "NULLABLE"},        # Primary key
                        #     {"name": "name_th", "type": "STRING", "mode": "NULLABLE"},
                        #     {"name": "name_en", "type": "STRING", "mode": "NULLABLE"},
                        #     {"name": "area_th", "type": "STRING", "mode": "NULLABLE"},
                        #     {"name": "area_en", "type": "STRING", "mode": "NULLABLE"},
                        #     {"name": "station_type", "type": "STRING", "mode": "NULLABLE"},
                        #     {"name": "lat", "type": "FLOAT", "mode": "NULLABLE"},                # Latitude
                        #     {"name": "long", "type": "FLOAT", "mode": "NULLABLE"},               # Longitude
                        #     {"name": "date", "type": "DATE", "mode": "NULLABLE"},                # Date field
                        #     {"name": "time", "type": "STRING", "mode": "NULLABLE"},                # Time field
                        #     {"name": "pm25_color_id", "type": "INTEGER", "mode": "NULLABLE"},    # Color ID
                        #     {"name": "pm25_aqi", "type": "INTEGER", "mode": "NULLABLE"},         # AQI Value
                        #     {"name": "pm25_value", "type": "FLOAT", "mode": "NULLABLE"},         # PM2.5 Value
                        #     {"name": "year", "type": "INTEGER", "mode": "NULLABLE"},             # Year
                        #     {"name": "month", "type": "INTEGER", "mode": "NULLABLE"},            # Month
                        #     {"name": "day", "type": "INTEGER", "mode": "NULLABLE"},   
                        #     {"name": "year", "type": "INTEGER", "mode": "NULLABLE"},
                        #     {"name": "month", "type": "INTEGER", "mode": "NULLABLE"},
                        #     {"name": "day", "type": "INTEGER", "mode": "NULLABLE"},              # Day
                        # ],  # เพิ่ม schema ของตาราง
                        "skipLeadingRows": 1,  # ข้าม header
                    }
                },
                gcp_conn_id="gcp_conn",
                execution_timeout=timedelta(minutes=30),
            )
            load_tasks_to_bigquery.append(load_task)
            print("Files in output folder:", os.listdir(output_folder))
        

    # Task สำหรับลงทะเบียนฟีเจอร์ใน Feast
    register_feature_task = PythonOperator(
        task_id="register_feature_in_feast",
        python_callable=register_feature_in_feast,
        execution_timeout=timedelta(minutes=30),
    )
    
    ingest_incremental_offline_task = PythonOperator(
    task_id="ingest_incremental_data",
    python_callable=ingest_incremental_data,
    dag=dag,
    )

    # กำหนดลำดับการทำงาน
    upload_task_to_gcs >> load_tasks_to_bigquery >> register_feature_task >> ingest_incremental_offline_task
