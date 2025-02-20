from airflow import DAG, Dataset
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta

# Feast
from feast import FeatureStore
from google.cloud import bigquery
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler


import sys
import os
import logging
import pandas as pd
import joblib
import numpy as np


# เพิ่ม path ไปยังโฟลเดอร์ที่มีไฟล์ feature_view_car_price.py
sys.path.append("/opt/airflow/feast")

# ✅ ใช้ Dataset URI ที่ถูกต้อง สำหรับเป็น Trigger
dataset_car_price = Dataset("file:///opt/airflow/dags/dataset_car_price/")

# ✅ ประกาศ Dataset ที่ใช้สำหรับ Trigger DAG อื่น
# dataset_transformed_car_price = Dataset("file:///opt/airflow/dags/dataset_transformed_car_price/")

default_args_transform_load = {
    'owner': 'airflow',
    'retries': 5,
    'retry_delay': timedelta(minutes=5),
}

data_folder = "/opt/airflow/dags/dataset_car_price"
transformed_folder = os.path.join(data_folder, 'transformed')

project_id = "is-my-project-428015"
dataset_id = "car_price_dataset"
table_id = "car_prices_non_feast"
table_id_1 = "car_price_predictions_non_feast"
table_id_2 = "car_ages_predictions_non_feast"

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/opt/airflow/credentials/is-my-project-428015-8661db9b13f2.json"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def feature_engineering():
    """
    อ่านไฟล์ CSV ดิบ, แปลง/เพิ่มคอลัมน์ (รวมถึง event_timestamp) 
    แล้วบันทึกไฟล์ CSV ที่ถูกแปลงแล้วไปยังโฟลเดอร์ transformed
    """
    if not os.path.exists(transformed_folder):
        os.makedirs(transformed_folder)

    transformed_files = set(os.listdir(transformed_folder))
    files_to_transform = [
        filename for filename in os.listdir(data_folder)
        if filename.endswith('.csv') and f"transformed_{filename}" not in transformed_files
    ]

    for filename in files_to_transform:
        try:
            file_path = os.path.join(data_folder, filename)
            df = pd.read_csv(file_path, on_bad_lines='skip', delimiter=',')

            # ปรับชื่อคอลัมน์เป็นตัวพิมพ์เล็ก
            df.columns = df.columns.str.lower()
            # ตัวอย่าง: เปลี่ยนชื่อคอลัมน์ doors -> car_doors
            df.rename(columns={'doors': 'car_doors'}, inplace=True)

            # เพิ่มฟีเจอร์ใหม่
            if 'price' in df.columns and 'mileage' in df.columns:
                df['price_per_mile'] = df['price'] / df['mileage']

            # เพิ่มคอลัมน์ event_timestamp ไว้เป็น timestamp สำหรับ Feast
            df['event_timestamp'] = pd.Timestamp.now()

            # บันทึกไฟล์ CSV ที่ถูกแปลง
            transformed_file_path = os.path.join(transformed_folder, f"transformed_{filename}")
            df.to_csv(transformed_file_path, index=False)
            logger.info(f"บันทึกไฟล์ที่แปลงแล้วสำเร็จ: {transformed_file_path}")
        except Exception as e:
            logger.error(f"เกิดข้อผิดพลาดกับไฟล์ {filename}: {e}")

def load_local_csv_to_bigquery():
    """
    นำไฟล์ CSV (ที่ผ่านการ Transform แล้ว) โหลดเข้า BigQuery 
    โดยกำหนด Schema ให้สอดคล้องกับฟีลด์ที่ต้องการ
    """
    client = bigquery.Client(project=project_id)

    if not os.path.exists(transformed_folder):
        logger.error(f"Folder {transformed_folder} does not exist.")
        return

    files = os.listdir(transformed_folder)
    csv_files = [f for f in files if f.endswith(".csv")]

    if not csv_files:
        logger.info("No CSV files found to load into BigQuery.")
        return

    for filename in csv_files:
        try:
            file_path = os.path.join(transformed_folder, filename)
            df = pd.read_csv(file_path)

            # แปลงคอลัมน์ event_timestamp ให้เป็น datetime
            df['event_timestamp'] = pd.to_datetime(df['event_timestamp'])

            table_ref = f"{project_id}.{dataset_id}.{table_id}"
            job_config = bigquery.LoadJobConfig(
                schema=[
                    bigquery.SchemaField("brand", "STRING"),
                    bigquery.SchemaField("model", "STRING"),
                    bigquery.SchemaField("year", "INTEGER"),
                    bigquery.SchemaField("engine_size", "FLOAT"),
                    bigquery.SchemaField("fuel_type", "STRING"),
                    bigquery.SchemaField("transmission", "STRING"),
                    bigquery.SchemaField("mileage", "INTEGER"),
                    bigquery.SchemaField("car_doors", "INTEGER"),
                    bigquery.SchemaField("owner_count", "INTEGER"),
                    bigquery.SchemaField("price", "INTEGER"),
                    bigquery.SchemaField("price_per_mile", "FLOAT"),
                    bigquery.SchemaField("event_timestamp", "TIMESTAMP"),
                ],
                write_disposition="WRITE_TRUNCATE"  # เขียนทับข้อมูลเดิม
            )
            job = client.load_table_from_dataframe(df, table_ref, job_config=job_config)
            job.result()

            logger.info(f"Loaded {filename} to BigQuery table {table_ref}")
        except Exception as e:
            logger.error(f"Error loading {filename} to BigQuery: {e}")


def train_model():
    """
    ดึงข้อมูลจาก BigQuery -> ทำ Feature Engineering -> Train โมเดล Random Forest -> บันทึกโมเดล
    """
    client = bigquery.Client(project=project_id)
    query = f"SELECT * FROM `{project_id}.{dataset_id}.{table_id}`"
    df = client.query(query).to_dataframe()
    
    # ✅ ตรวจสอบว่ามีโฟลเดอร์ models หรือไม่ ถ้าไม่มีให้สร้าง
    models_folder = "/opt/airflow/models"
    if not os.path.exists(models_folder):
        os.makedirs(models_folder)
    
    # Feature Engineering car_price
    df["car_age"] = 2025 - df["year"]
    df["mileage_per_year"] = (df["mileage"] / (df["car_age"] + 1)).round(2)
    
    categorical_features = ["brand", "model", "fuel_type", "transmission"]
    label_encoders = {}
    for col in categorical_features:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le
    
    features = ["brand", "model", "car_age", "engine_size", "fuel_type", "transmission", "mileage", "mileage_per_year", "car_doors", "owner_count"]
    target = "price" 
    
    X = df[features]
    y = df[target]   

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_scaled, y)
    
    
    joblib.dump(model, "/opt/airflow/models/car_price_model.pkl")
    joblib.dump(scaler, "/opt/airflow/models/scaler.pkl")
    joblib.dump(label_encoders, "/opt/airflow/models/label_encoders.pkl")
    
    logger.info("✅ Model trained and saved!")



# def train_model_car_age():
#     """
#     ดึงข้อมูลจาก BigQuery -> ทำ Feature Engineering -> Train โมเดล Random Forest -> บันทึกโมเดล
#     """
#     client = bigquery.Client(project=project_id)
#     query = f"SELECT * FROM `{project_id}.{dataset_id}.{table_id}`"
#     df = client.query(query).to_dataframe()
    
#     # ✅ ตรวจสอบว่ามีโฟลเดอร์ models หรือไม่ ถ้าไม่มีให้สร้าง
#     models_folder = "/opt/airflow/models"
#     if not os.path.exists(models_folder):
#         os.makedirs(models_folder)
        
#     # Feature Engineering car_ages
#     df["car_age"] = 2025 - df["year"]
#     df["mileage_per_year"] = (df["mileage"] / (df["car_age"] + 1)).round(2)
#     df["remaining_useful_life"] = 20 - df["car_age"]  # สมมติอายุการใช้งานเฉลี่ย 20 ปี
#     df["remaining_useful_life"] = df["remaining_useful_life"].clip(lower=0)
    
#     categorical_features = ["brand", "model", "fuel_type", "transmission"]
#     label_encoders = {}
#     for col in categorical_features:
#         le = LabelEncoder()
#         df[col] = le.fit_transform(df[col])
#         label_encoders[col] = le
    
#     features = ["brand", "model", "car_age", "engine_size", "fuel_type", "transmission", "mileage", "mileage_per_year", "car_doors", "owner_count"]
#     target = "remaining_useful_life"
    
#     X = df[features]
#     y = df[target]   
    
#     scaler = StandardScaler()
#     X_scaled = scaler.fit_transform(X)
    
#     model = RandomForestRegressor(n_estimators=100, random_state=42)
#     model.fit(X_scaled, y)
    
#     joblib.dump(model, os.path.join(models_folder, "car_ages_model.pkl"))
#     joblib.dump(scaler, os.path.join(models_folder, "scaler.pkl"))
#     joblib.dump(label_encoders, os.path.join(models_folder, "label_encoders.pkl"))
    
#     logger.info("✅ car_ages Model trained and saved!")


def predict_car_prices():
    """
    โหลดโมเดล -> ดึงข้อมูลใหม่จาก BigQuery -> ทำนายราคาขาย -> บันทึกผลลัพธ์
    """
    client = bigquery.Client(project=project_id)
    query = f"SELECT * FROM `{project_id}.{dataset_id}.{table_id}`"
    df = client.query(query).to_dataframe()
    
    # ✅ โหลดโมเดลที่ Train แล้ว
    model = joblib.load("/opt/airflow/models/car_price_model.pkl")
    scaler = joblib.load("/opt/airflow/models/scaler.pkl")
    label_encoders = joblib.load("/opt/airflow/models/label_encoders.pkl")
    
    # สร้างคอลัมน์ car_age
    # สร้างคอลัมน์ mileage_per_year
    df["car_age"] = 2025 - df["year"]
    df["mileage_per_year"] = (df["mileage"] / (df["car_age"] + 1)).round(2)
    
    categorical_features = ["brand", "model", "fuel_type", "transmission"]
    for col in categorical_features:
        df[col] = label_encoders[col].transform(df[col])

    features = ["brand", "model", "car_age", "engine_size", "fuel_type", "transmission", "mileage", "mileage_per_year", "car_doors", "owner_count"]
    X = df[features]
    X_scaled = scaler.transform(X)
    
    df["predicted_price"] = model.predict(X_scaled)

    # ✅ แปลงค่าตัวเลขกลับเป็นชื่อจริง
    df["brand_name"] = label_encoders["brand"].inverse_transform(df["brand"])
    df["model_name"] = label_encoders["model"].inverse_transform(df["model"])
    
    # ✅ บันทึกผลลัพธ์ลง BigQuery
    table_ref = f"{project_id}.{dataset_id}.{table_id_1}"
    job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    
    job = client.load_table_from_dataframe(
        df[["brand", "brand_name", "model", "model_name", "year", "mileage", "mileage_per_year", "predicted_price"]],
        table_ref,
        job_config=job_config
    )
    job.result()
    
    logger.info(f"✅ Predictions saved to BigQuery table {table_ref}")


# def predict_car_age():
#     """
#     โหลดโมเดล -> ดึงข้อมูลใหม่จาก BigQuery -> ทำนาย RUL -> บันทึกผลลัพธ์
#     """
    
#     models_folder = "/opt/airflow/models"
#     client = bigquery.Client(project=project_id)
#     query = f"SELECT * FROM `{project_id}.{dataset_id}.{table_id}`"
#     df = client.query(query).to_dataframe()
    
#     # ตรวจสอบคอลัมน์ใน DataFrame
#     print("Columns in DataFrame:", df.columns.tolist())
    
#     # โหลดโมเดลที่ Train แล้ว
#     model = joblib.load(os.path.join(models_folder, "car_ages_model.pkl"))
#     scaler = joblib.load(os.path.join(models_folder, "scaler.pkl"))
#     label_encoders = joblib.load(os.path.join(models_folder, "label_encoders.pkl"))
    
#     df["car_age"] = 2025 - df["year"]
#     df["mileage_per_year"] = (df["mileage"] / (df["car_age"] + 1)).round(2)
    
#     categorical_features = ["brand", "model", "fuel_type", "transmission"]
#     for col in categorical_features:
#         df[col] = label_encoders[col].transform(df[col])
    
#     features = ["brand", "model", "car_age", "engine_size", "fuel_type", "transmission", "mileage", "mileage_per_year", "car_doors", "owner_count"]
#     X = df[features]
#     X_scaled = scaler.transform(X)
    
#     df["predicted_car_age"] = model.predict(X_scaled)
    
#     # ✅ แปลงค่าตัวเลขกลับเป็นชื่อจริง
#     df["brand_name"] = label_encoders["brand"].inverse_transform(df["brand"])
#     df["model_name"] = label_encoders["model"].inverse_transform(df["model"])    
    
#     # บันทึกผลลัพธ์ลง BigQuery
#     table_ref = f"{project_id}.{dataset_id}.{table_id_2}"
#     job_config = bigquery.LoadJobConfig(write_disposition="WRITE_TRUNCATE")
    
#     job = client.load_table_from_dataframe(
#         df[["brand", "brand_name", "model", "model_name", "year", "mileage", "predicted_car_age"]],
#         table_ref,
#         job_config=job_config
#     )
#     job.result()
    
#     logger.info(f"✅ RUL Predictions saved to BigQuery table {table_ref}")


with DAG(
    dag_id="non_feast_transform_and_load_car_price_data",
    default_args=default_args_transform_load,
    schedule=[dataset_car_price],  # Dataset เป็น Trigger
    start_date=datetime(2025, 2, 3),
    catchup=False,
    tags=["swu"],
) as dag_transform_and_load:

    transform_data_task = PythonOperator(
        task_id='feature_engineering',
        python_callable=feature_engineering,
        execution_timeout=timedelta(minutes=30),
    )

    load_to_bigquery_task = PythonOperator(
        task_id='load_local_csv_to_bigquery',
        python_callable=load_local_csv_to_bigquery,
        execution_timeout=timedelta(minutes=30),
    )
    
    training_model = PythonOperator(
        task_id="train_model",
        python_callable=train_model,
        execution_timeout=timedelta(minutes=30),
    )
    
    # training_model_car_age = PythonOperator(
    #     task_id="train_model_car_age",
    #     python_callable=train_model_car_age,
    #     execution_timeout=timedelta(minutes=30),
    # ) 
    
    prediction_prices = PythonOperator(
        task_id="predict_car_prices",
        python_callable=predict_car_prices,
        execution_timeout=timedelta(minutes=30),
    ) 

    # predict_car_ages = PythonOperator(
    #     task_id="predict_car_age",
    #     python_callable=predict_car_age,
    #     execution_timeout=timedelta(minutes=30),
    # ) 
  
    transform_data_task >> load_to_bigquery_task
    load_to_bigquery_task >> training_model >> prediction_prices
    # load_to_bigquery_task >> training_model_car_age >> predict_car_ages















# from airflow import DAG, Dataset
# from airflow.operators.python_operator import PythonOperator
# from datetime import datetime, timedelta
# import os
# import logging

# # PySpark
# from pyspark.sql import SparkSession
# from pyspark.sql.functions import (
#     col, lit, when, current_timestamp, expr
# )
# from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType, TimestampType

# # Spark ML
# from pyspark.ml import Pipeline, PipelineModel
# from pyspark.ml.feature import (
#     StringIndexer,
#     VectorAssembler
# )
# from pyspark.ml.regression import RandomForestRegressor
# from pyspark.sql.utils import AnalysisException

# # ✅ ใช้ Dataset URI ที่ถูกต้อง สำหรับเป็น Trigger
# dataset_car_price = Dataset("file:///opt/airflow/dags/dataset_car_price/")

# # ตั้งค่าพื้นฐานของ DAG
# default_args_transform_load = {
#     'owner': 'airflow',
#     'retries': 5,
#     'retry_delay': timedelta(minutes=5),
# }

# data_folder = "/opt/airflow/dags/dataset_car_price"
# transformed_folder = os.path.join(data_folder, 'transformed')

# project_id = "is-my-project-428015"
# dataset_id = "car_price_dataset"
# table_id = "spark_car_prices"          # ตารางเก็บข้อมูลหลัง transform
# table_id_1 = "spark_car_price_predictions"  # ตารางเก็บผลการพยากรราคา
# table_id_2 = "spark_car_ages_predictions"   # ตารางเก็บผลการพยากรอายุ

# # ไฟล์ Credential (ถ้าจำเป็นต้องตั้งค่าเพิ่มใน SparkSession)
# os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/opt/airflow/credentials/is-my-project-428015-8661db9b13f2.json"

# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


# def feature_engineering():
#     """
#     อ่านไฟล์ CSV ดิบด้วย Spark, แปลง/เพิ่มคอลัมน์ (event_timestamp), 
#     แล้วบันทึก CSV ที่ถูกแปลงแล้วไปยังโฟลเดอร์ transformed
#     """
#     spark = SparkSession.builder.appName("feature_engineering").getOrCreate()

#     if not os.path.exists(transformed_folder):
#         os.makedirs(transformed_folder)

#     transformed_files = set(os.listdir(transformed_folder))
#     files_to_transform = [
#         filename for filename in os.listdir(data_folder)
#         if filename.endswith('.csv') and f"transformed_{filename}" not in transformed_files
#     ]

#     for filename in files_to_transform:
#         try:
#             file_path = os.path.join(data_folder, filename)
#             # อ่านไฟล์ CSV
#             df = spark.read.csv(file_path, header=True, inferSchema=True)
            
#             # เปลี่ยนชื่อคอลัมน์เป็นตัวพิมพ์เล็กทั้งหมด
#             for c in df.columns:
#                 df = df.withColumnRenamed(c, c.lower())
            
#             # เปลี่ยนชื่อคอลัมน์ doors -> car_doors (ถ้ามีอยู่)
#             if 'doors' in df.columns:
#                 df = df.withColumnRenamed('doors', 'car_doors')
            
#             # เพิ่มฟีเจอร์ใหม่ price_per_mile (ถ้ามี price และ mileage)
#             if 'price' in df.columns and 'mileage' in df.columns:
#                 df = df.withColumn('price_per_mile', col('price') / col('mileage'))
            
#             # เพิ่มคอลัมน์ event_timestamp
#             df = df.withColumn('event_timestamp', current_timestamp())
            
#             # บันทึกไฟล์ CSV ที่ถูกแปลง
#             transformed_file_path = os.path.join(transformed_folder, f"transformed_{filename}")
#             # โค้ด spark.write.csv จะสร้างโฟลเดอร์แทนไฟล์เดียว
#             # สำหรับความง่าย อาจใช้ coalesce(1) บังคับให้มีไฟล์เดียว แต่การ production จริงอาจไม่เหมาะ
#             df.coalesce(1).write.csv(transformed_file_path, header=True, mode='overwrite')
#             logger.info(f"บันทึกไฟล์ที่แปลงแล้วสำเร็จ: {transformed_file_path}")
#         except Exception as e:
#             logger.error(f"เกิดข้อผิดพลาดกับไฟล์ {filename}: {e}")

#     spark.stop()


# def load_local_csv_to_bigquery():
#     """
#     นำไฟล์ CSV (ที่ผ่านการ Transform แล้ว) โหลดเข้า BigQuery
#     ใช้ Spark BigQuery Connector
#     """
#     spark = SparkSession.builder \
#         .appName("load_to_bigquery") \
#         .config("spark.jars.packages",
#                 "com.google.cloud.spark:spark-bigquery-with-dependencies_2.12:0.36.1") \
#         .config("spark.hadoop.fs.gs.impl", "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem") \
#         .config("spark.hadoop.fs.AbstractFileSystem.gs.impl", "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFS") \
#         .config("spark.executor.memory", "2g") \
#         .config("spark.driver.memory", "2g") \
#         .getOrCreate()

#     if not os.path.exists(transformed_folder):
#         logger.error(f"Folder {transformed_folder} does not exist.")
#         spark.stop()
#         return

#     csv_files = [
#         f for f in os.listdir(transformed_folder)
#         if f.endswith(".csv") or f.endswith(".csv.gz")  # เผื่อกรณีบีบอัด
#     ]

#     if not csv_files:
#         logger.info("No CSV files found to load into BigQuery.")
#         spark.stop()
#         return

#     for filename in csv_files:
#         try:
#             file_path = os.path.join(transformed_folder, filename)
#             df = spark.read.csv(path=f"file://{file_path}", header=True, inferSchema=True)

#             # ตรวจสอบคอลัมน์ที่ต้องการ (กำหนด schema แบบง่าย ๆ)
#             # ถ้าต้องกำหนด schema ละเอียดกว่า อาจใช้ StructType แทน inferSchema
#             # แปลงคอลัมน์ event_timestamp ให้เป็น TimestampType ถ้าไม่ใช่
#             df = df.withColumn("event_timestamp", col("event_timestamp").cast(TimestampType()))

#             # ใส่ .option("temporaryGcsBucket", "gear-bucket-gcs") เพื่อให้ BQ Connector ใช้เป็น staging bucket
#             df.write \
#               .format("bigquery") \
#               .option("table", f"{project_id}:{dataset_id}.{table_id}") \
#               .option("temporaryGcsBucket", "gear-bucket-gcs") \
#               .mode("append") \
#               .save()

#             # # เขียนลง BigQuery (mode="overwrite" เทียบเท่า WRITE_TRUNCATE)
#             # df.write \
#             #   .format("bigquery") \
#             #   .option("table", f"{project_id}:{dataset_id}.{table_id}") \
#             #   .mode("append") \
#             #   .save()

#             logger.info(f"Loaded {filename} to BigQuery table {project_id}.{dataset_id}.{table_id}")
#         except Exception as e:
#             logger.error(f"Error loading {filename} to BigQuery: {e}")

#     spark.stop()


# def train_model():
#     """
#     ดึงข้อมูลจาก BigQuery -> ทำ Feature Engineering ใน Spark -> Train โมเดล RandomForest (Spark ML) -> บันทึกโมเดล
#     """
#     spark = SparkSession.builder.appName("train_car_price_model").getOrCreate()
#     try:
#         # อ่านข้อมูลจาก BigQuery
#         df = spark.read.format("bigquery") \
#             .option("table", f"{project_id}:{dataset_id}.{table_id}") \
#             .load()

#         # สร้างคอลัมน์ car_age, mileage_per_year
#         df = df.withColumn("car_age",  expr("2025 - year"))
#         df = df.withColumn("mileage_per_year", col("mileage") / (col("car_age") + lit(1)))

#         # แปลงค่าข้อความ -> ตัวเลขด้วย StringIndexer
#         cat_cols = ["brand", "model", "fuel_type", "transmission"]
#         indexers = [
#             StringIndexer(inputCol=c, outputCol=c+"_idx", handleInvalid="keep")
#             for c in cat_cols
#         ]

#         # รวมฟีเจอร์
#         feature_cols = [
#             "brand_idx", "model_idx", "car_age", "engine_size",
#             "fuel_type_idx", "transmission_idx", "mileage",
#             "mileage_per_year", "car_doors", "owner_count"
#         ]
#         assembler = VectorAssembler(
#             inputCols=feature_cols,
#             outputCol="features"
#         )

#         # ตั้งค่า RandomForestRegressor
#         rf = RandomForestRegressor(
#             featuresCol="features",
#             labelCol="price",  # กำหนด target เป็น price
#             numTrees=100,
#             maxDepth=10,
#             seed=42
#         )

#         # สร้าง Pipeline
#         pipeline = Pipeline(stages=indexers + [assembler, rf])

#         # เทรนโมเดล
#         model = pipeline.fit(df)
        
#         # บันทึกโมเดล (Spark ML จะบันทึกเป็นโฟลเดอร์)
#         models_folder = "/opt/airflow/models_spark"
#         if not os.path.exists(models_folder):
#             os.makedirs(models_folder)

#         model.save(os.path.join(models_folder, "car_price_model_spark"))
#         logger.info("✅ Spark ML Model (car_price) trained and saved!")
#     except AnalysisException as e:
#         logger.error(f"Error reading from BigQuery: {e}")
#     except Exception as ex:
#         logger.error(f"Train model error: {ex}")
#     finally:
#         spark.stop()


# def train_model_car_age():
#     """
#     ดึงข้อมูลจาก BigQuery -> ทำ Feature Engineering -> Train โมเดล RandomForest (Spark ML) -> บันทึกโมเดล
#     สำหรับทำนาย remaining_useful_life
#     """
#     spark = SparkSession.builder.appName("train_car_age_model").getOrCreate()
#     try:
#         df = spark.read.format("bigquery") \
#             .option("table", f"{project_id}:{dataset_id}.{table_id}") \
#             .load()

#         # สร้างคอลัมน์ car_age, mileage_per_year, remaining_useful_life
#         df = df.withColumn("car_age",  expr("2025 - year"))
#         df = df.withColumn("mileage_per_year", col("mileage") / (col("car_age") + lit(1)))
#         df = df.withColumn("remaining_useful_life", expr("20 - car_age"))
#         df = df.withColumn("remaining_useful_life", when(col("remaining_useful_life") < 0, 0).otherwise(col("remaining_useful_life")))

#         cat_cols = ["brand", "model", "fuel_type", "transmission"]
#         indexers = [
#             StringIndexer(inputCol=c, outputCol=c+"_idx", handleInvalid="keep")
#             for c in cat_cols
#         ]

#         feature_cols = [
#             "brand_idx", "model_idx", "car_age", "engine_size",
#             "fuel_type_idx", "transmission_idx", "mileage",
#             "mileage_per_year", "car_doors", "owner_count"
#         ]
#         assembler = VectorAssembler(
#             inputCols=feature_cols,
#             outputCol="features"
#         )

#         rf = RandomForestRegressor(
#             featuresCol="features",
#             labelCol="remaining_useful_life",
#             numTrees=100,
#             maxDepth=10,
#             seed=42
#         )

#         pipeline = Pipeline(stages=indexers + [assembler, rf])
#         model = pipeline.fit(df)

#         models_folder = "/opt/airflow/models_spark"
#         if not os.path.exists(models_folder):
#             os.makedirs(models_folder)

#         model.save(os.path.join(models_folder, "car_age_model_spark"))
#         logger.info("✅ Spark ML Model (car_age) trained and saved!")
#     except AnalysisException as e:
#         logger.error(f"Error reading from BigQuery: {e}")
#     except Exception as ex:
#         logger.error(f"Train car age model error: {ex}")
#     finally:
#         spark.stop()


# def predict_car_prices():
#     """
#     โหลดโมเดล Spark ML -> ดึงข้อมูลใหม่จาก BigQuery -> ทำนายราคา -> บันทึกผลลัพธ์
#     """
#     spark = SparkSession.builder.appName("predict_car_prices").getOrCreate()
#     try:
#         df = spark.read.format("bigquery") \
#             .option("table", f"{project_id}:{dataset_id}.{table_id}") \
#             .load()

#         # โหลดโมเดล
#         models_folder = "/opt/airflow/models_spark"
#         model_path = os.path.join(models_folder, "car_price_model_spark")
#         model = PipelineModel.load(model_path)

#         # ทำ inference
#         predictions = model.transform(df)

#         # ดึงเฉพาะคอลัมน์ที่ต้องการ
#         # col("prediction") คือ "predicted_price"
#         result = predictions.select(
#             "brand", "model", "year", "mileage",
#             col("prediction").alias("predicted_price")
#         )

#         # เขียนลง BigQuery (write_disposition = "WRITE_TRUNCATE" => ใช้ mode="overwrite")
#         result.write \
#               .format("bigquery") \
#               .option("table", f"{project_id}:{dataset_id}.{table_id_1}") \
#               .mode("overwrite") \
#               .save()

#         logger.info(f"✅ Predictions saved to BigQuery table {project_id}.{dataset_id}.{table_id_1}")
#     except AnalysisException as e:
#         logger.error(f"Error reading from BigQuery: {e}")
#     except Exception as ex:
#         logger.error(f"Predict car prices error: {ex}")
#     finally:
#         spark.stop()


# def predict_car_age():
#     """
#     โหลดโมเดล Spark ML -> ดึงข้อมูลใหม่จาก BigQuery -> ทำนาย RUL -> บันทึกผลลัพธ์
#     """
#     spark = SparkSession.builder.appName("predict_car_age").getOrCreate()
#     try:
#         df = spark.read.format("bigquery") \
#             .option("table", f"{project_id}:{dataset_id}.{table_id}") \
#             .load()

#         models_folder = "/opt/airflow/models_spark"
#         model_path = os.path.join(models_folder, "car_age_model_spark")
#         model = PipelineModel.load(model_path)

#         predictions = model.transform(df)
#         result = predictions.select(
#             "brand", "model", "year", "mileage",
#             col("prediction").alias("predicted_car_age")
#         )

#         result.write \
#               .format("bigquery") \
#               .option("table", f"{project_id}:{dataset_id}.{table_id_2}") \
#               .mode("overwrite") \
#               .save()

#         logger.info(f"✅ RUL Predictions saved to BigQuery table {project_id}.{dataset_id}.{table_id_2}")
#     except AnalysisException as e:
#         logger.error(f"Error reading from BigQuery: {e}")
#     except Exception as ex:
#         logger.error(f"Predict car age error: {ex}")
#     finally:
#         spark.stop()


# # ---------------------------------------------------
# # กำหนด Airflow DAG และ Task
# # ---------------------------------------------------
# with DAG(
#     dag_id="non_feast_transform_and_load_car_price_data_spark",
#     default_args=default_args_transform_load,
#     schedule=[dataset_car_price],  # Trigger จาก Dataset
#     start_date=datetime(2025, 2, 3),
#     catchup=False,
#     tags=["swu"],
# ) as dag_transform_and_load:

#     transform_data_task = PythonOperator(
#         task_id='feature_engineering',
#         python_callable=feature_engineering,
#         execution_timeout=timedelta(minutes=30),
#     )

#     load_to_bigquery_task = PythonOperator(
#         task_id='load_local_csv_to_bigquery',
#         python_callable=load_local_csv_to_bigquery,
#         execution_timeout=timedelta(minutes=30),
#     )
    
#     training_model = PythonOperator(
#         task_id="train_model",
#         python_callable=train_model,
#         execution_timeout=timedelta(minutes=30),
#     )
    
#     training_model_car_age = PythonOperator(
#         task_id="train_model_car_age",
#         python_callable=train_model_car_age,
#         execution_timeout=timedelta(minutes=30),
#     ) 
    
#     prediction_prices = PythonOperator(
#         task_id="predict_car_prices",
#         python_callable=predict_car_prices,
#         execution_timeout=timedelta(minutes=30),
#     ) 

#     predict_car_ages = PythonOperator(
#         task_id="predict_car_age",
#         python_callable=predict_car_age,
#         execution_timeout=timedelta(minutes=30),
#     ) 
  
#     # กำหนดลำดับการรัน
#     transform_data_task >>load_to_bigquery_task
#     load_to_bigquery_task >> training_model >> prediction_prices
#     load_to_bigquery_task >> training_model_car_age >> predict_car_ages



















