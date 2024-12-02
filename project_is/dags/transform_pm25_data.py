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
    'retries': 10,
    'retry_delay': timedelta(minutes=5),
}

def feature_engineering():
    if not os.path.exists(transformed_folder):
        os.makedirs(transformed_folder)
        
    transformed_files = set(os.listdir(transformed_folder))

    files_to_transform = [
        filename for filename in os.listdir(data_folder)
        if filename.endswith('.csv') and f"transformed_{filename}" not in transformed_files
    ]

    # นิยามฟังก์ชัน categorize_pm25 ภายใน feature_engineering
    def categorize_pm25(pm25_color_id):
        if pm25_color_id < 2:
            return 'Very Low'
        elif 2 <= pm25_color_id < 3:
            return 'Moderate'
        elif 3 <= pm25_color_id < 4:
            return 'High'
        else:
            return 'Very High'
        
        
    for filename in files_to_transform:
            try:
                file_path = os.path.join(data_folder, filename)
                df = pd.read_csv(file_path, on_bad_lines='skip', delimiter=',')
                
                df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d', errors='coerce')
                df.dropna(subset=['date'], inplace=True)  # ลบบรรทัดที่แปลง date ไม่ได้
                df['year'] = df['date'].dt.year
                df['month'] = df['date'].dt.month
                df['day'] = df['date'].dt.day
                
                # แปลงข้อมูล PM2.5 โดยใช้ฟังก์ชัน categorize_pm25
                if 'pm25_color_id' in df.columns:
                    df['pm25_category'] = df['pm25_color_id'].apply(categorize_pm25)
                else:
                    print(f"ไม่มีคอลัมน์ 'pm25_color_id' ในไฟล์ {filename}")
                
                # บันทึกไฟล์ที่แปลงแล้ว
                transformed_file_path = os.path.join(transformed_folder, f"transformed_{filename}")
                df.to_csv(transformed_file_path, index=False)
                print(f"บันทึกไฟล์ที่แปลงแล้วสำเร็จ: {transformed_file_path}")
            except Exception as e:
                print(f"เกิดข้อผิดพลาดกับไฟล์ {filename}: {e}")
            

def get_files_to_load():
    # ส่งเฉพาะไฟล์ที่ยังไม่ได้โหลดไปยัง DAG อื่น
    files_to_transform = os.listdir(data_folder)
    transformed_files = os.listdir(transformed_folder)
    files_ready_to_load = [
        f"transformed_{filename}" for filename in files_to_transform
        if f"transformed_{filename}" in transformed_files
    ]
    return files_ready_to_load   
            

with DAG('transform_pm25_data',
         default_args=default_args,
         schedule_interval=None,
         start_date=datetime(2024, 11, 3),
         tags=['swu'],
         catchup=False) as dag:


    transform_data_task = PythonOperator(
        task_id='feature_engineering',
        python_callable=feature_engineering,
        execution_timeout=timedelta(minutes=30),
    )

    def trigger_load_dag():
        # ส่งไฟล์ที่พร้อมโหลดไปยัง DAG load_pm25_feature_store
        files_to_load = get_files_to_load()
        if files_to_load:
            print(f"ไฟล์ที่พร้อมส่งไปยัง DAG อื่น: {files_to_load}")
            # โค้ดสำหรับส่งไฟล์ไปยัง DAG อื่น (สามารถปรับตามต้องการ)
        else:
            print("ไม่มีไฟล์ที่ต้องส่งไปยัง DAG อื่น")

    trigger_load_data_task = PythonOperator(
        task_id='trigger_load_to_bigquery',
        python_callable=trigger_load_dag,
        execution_timeout=timedelta(minutes=10),
    )
    
    trigger_next_load_pm25_feature_store = TriggerDagRunOperator(
       task_id='upload_task_to_gcs',
       trigger_dag_id='load_pm25_feature_store',
       execution_timeout=timedelta(minutes=10),
    )

transform_data_task >> trigger_load_data_task >> trigger_next_load_pm25_feature_store
