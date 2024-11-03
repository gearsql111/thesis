from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from datetime import datetime, timedelta
import pandas as pd
import os

# กำหนดที่อยู่โฟลเดอร์สำหรับไฟล์ที่ได้จากขั้นตอนดึงข้อมูล และที่อยู่โฟลเดอร์ใหม่สำหรับเก็บไฟล์ที่ผ่านการแปลงแล้ว
data_folder = '/opt/airflow/dags/data_test/'  # ที่อยู่ข้อมูลจาก DAG ก่อนหน้า
transformed_folder = data_folder + 'transformed/'

default_args = {
    'owner': 'airflow',
    'retries': 5,
    'retry_delay': timedelta(minutes=5),
}

def feature_engineering():
    # ตรวจสอบและสร้างโฟลเดอร์สำหรับเก็บไฟล์ที่ผ่านการแปลง
    if not os.path.exists(transformed_folder):
        os.makedirs(transformed_folder)

    # อ่านไฟล์ที่ดาวน์โหลดมาและทำการแปลงข้อมูล
    for filename in os.listdir(data_folder):
        if filename.endswith('.csv'):
            # โหลดข้อมูลจากไฟล์ CSV
            df = pd.read_csv(os.path.join(data_folder, filename))
            
            # ตัวอย่างการทำ Feature Engineering
             # 1. สร้าง Feature ด้านเวลา: ปี, เดือน, วัน
            df['date'] = pd.to_datetime(df['date'], format='%d/%m/%Y')
            df['year'] = df['date'].dt.year
            df['month'] = df['date'].dt.month
            df['day'] = df['date'].dt.day
            
            # # 2. เพิ่มฟีเจอร์ใหม่ เช่น ช่วงเวลา (เช้า, บ่าย, เย็น, กลางคืน)
            # df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
            # df['time_of_day'] = df['hour'].apply(lambda x: 'morning' if 6 <= x < 12 else
            #                                                 'afternoon' if 12 <= x < 18 else
            #                                                 'evening' if 18 <= x < 24 else 'night')
            
            # 3. การจัดระดับค่า PM2.5 และ AQI เป็นกลุ่ม: กลุ่ม PM2.5 และ AQI
            # สมมติว่าเรามีคอลัมน์ 'pm25' และ 'aqi' ในข้อมูล
            def categorize_pm25(pm25_color_id):
                if pm25_color_id < 2:
                    return 'Very Low'
                elif 2 <= pm25_color_id < 3:
                    return 'Moderate'
                elif 3 <= pm25_color_id < 4:
                    return 'High'
                else:
                    return 'Very High'
            
            # 3. แปลงข้อมูลจากหน่วยเดิมให้เป็นค่ามาตรฐาน
            # df['pm25_standard'] = df['pm25'] / 25.0  # ตัวอย่างการ normalize ข้อมูล
            
            # บันทึกไฟล์ที่แปลงแล้วในโฟลเดอร์ใหม่
            transformed_file_path = os.path.join(transformed_folder, 'transformed_' + filename)
            df.to_csv(transformed_file_path, index=False)
            print(f"บันทึกไฟล์ที่แปลงแล้วสำเร็จ: {transformed_file_path}")

with DAG('transform_pm25_data',
         default_args=default_args,
         schedule_interval=None,
         start_date=datetime(2024, 11, 3),
         catchup=False) as dag:

    # Task สำหรับทำ Feature Engineering
    transform_data_task = PythonOperator(
        task_id='feature_engineering',
        python_callable=feature_engineering,
    )

    # Trigger ไปยัง DAG ถัดไปที่ทำการโหลดข้อมูลขึ้น BigQuery
    trigger_load_data_task = TriggerDagRunOperator(
        task_id='trigger_load_data_to_bigquery',
        trigger_dag_id='load_pm25_feature_store',  # DAG ถัดไป
    )

transform_data_task >> trigger_load_data_task