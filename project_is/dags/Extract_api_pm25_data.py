from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
from airflow.operators.empty import EmptyOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from bs4 import BeautifulSoup
from urllib.parse import urlparse


import requests
import os

default_args = {
    'owner': 'airflow',
    'retries': 5,
    'retry_delay': timedelta(minutes=5),
}

data_path = '/opt/airflow/dags/data_test'
data_folder = data_path + '/'
clean_folder = data_folder + 'cleaned'


def fetch_data():
    # check ว่ามี folder หรือไม่ ถ้าไม่มีก็สร้าง folder ใหม่
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    # check ที่เก็บอยู่ในตัวแปร data_folder มีอยู่หรือไม่.
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)

    # if not os.path.exists(clean_folder):
    #     os.makedirs(clean_folder)

    # ดึง files ทั้งหมดที่อยู่ใน folder data_folder
    exit_file_name = os.listdir(data_folder)
    url = 'https://opendata.onde.go.th/dataset/14-pm-25'
    links = []  # เอาไว้เก็บค่าต่าง ๆ ที่ถูกดึงมาจาก URL

    req = requests.get(url, verify=False)
    req.encoding = 'utf-8'
    # การแปลง HTML เป็นโครงสร้างข้อมูล
    soup = BeautifulSoup(req.text, 'html.parser')

    og = soup.find('meta', property='og:url')
    base = urlparse(url)

    # loop ข้อมูลทั้งหมดที่มีอยู่ในหน้าเว็บที่แปลงเป็นโครงสร้างข้อมูล
    for link in soup.find_all('a'):
        current_link = link.get('href')  # ดึงค่าจาก a
        if str(current_link).endswith('csv'):  # check current_link ลงท้ายด้วย .csv รึป่าว
            links.append(current_link)

    for link in links:  # loop แต่ละตัวที่เก็บใน links และแยก url เพื่อเก็บชื่อตัวสุดท้าย
        names = link.split('/')[-1]
        names = names.strip()  # ลบช่องว่าง (whitespace)
        name = names.replace('pm_data_hourly-', '')

        # ใช้ check ไฟล์ที่ไม่ต้องการให้ดาวน์โหลดซ้ำ
        if (name != 'data_dictionary.csv') & (name not in exit_file_name):
            req = requests.get(link, verify=False)
            url_content = req.content
            file_path = data_folder + name
            # บันทึกไฟล์ CSV
            with open(file_path, 'wb') as csv_file:
                csv_file.write(req.content)
            print(f"ดาวน์โหลดไฟล์ CSV สำเร็จ: {name}")
    else:
        print(f"ไม่สามารถเข้าถึงหน้าเว็บ: {req.status_code}")


with DAG('Extract_api_pm25_data', 
         default_args=default_args, 
         schedule_interval=None, 
         start_date=datetime(2024, 11, 3), 
         tags=['swu'],
         catchup=False) as dag:
    
    start_task = EmptyOperator(
        task_id='start',
    )
    
    fetch_data_task = PythonOperator(
        task_id='fetch_data',
        python_callable=fetch_data,
    )
    
    trigger_next_transform_pm25_data = TriggerDagRunOperator(
        task_id='feature_engineering_task',
        trigger_dag_id='transform_pm25_data',  # ID ของ DAG ถัดไป
    )

start_task >> fetch_data_task >> trigger_next_transform_pm25_data