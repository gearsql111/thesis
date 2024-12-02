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
    'retries': 10,
    'retry_delay': timedelta(minutes=5),
}

data_path = '/opt/airflow/dags/data_test'
data_folder = data_path + '/'
clean_folder = data_folder + 'cleaned'


def fetch_data():
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    if not os.path.exists(data_folder):
        os.makedirs(data_folder)


    exit_file_name = os.listdir(data_folder)
    url = 'https://opendata.onde.go.th/dataset/14-pm-25'
    links = []

    req = requests.get(url, verify=False)
    req.encoding = 'utf-8'

    soup = BeautifulSoup(req.text, 'html.parser')

    og = soup.find('meta', property='og:url')
    base = urlparse(url)


    for link in soup.find_all('a'):
        current_link = link.get('href') 
        if str(current_link).endswith('csv'): 
            links.append(current_link)

    for link in links:
        names = link.split('/')[-1]
        names = names.strip()
        name = names.replace('pm_data_hourly-', '')


        if (name != 'data_dictionary.csv') & (name not in exit_file_name):
            req = requests.get(link, verify=False)
            url_content = req.content
            file_path = data_folder + name

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
        execution_timeout=timedelta(minutes=30),
    )
    
    trigger_next_transform_pm25_data = TriggerDagRunOperator(
        task_id='feature_engineering',
        trigger_dag_id='transform_pm25_data',
        execution_timeout=timedelta(minutes=10),
    )

start_task >> fetch_data_task >> trigger_next_transform_pm25_data