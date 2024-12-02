# เริ่มต้นจาก Airflow image ที่รองรับ Python 3.10 โดยตรง
FROM apache/airflow:2.10.3-python3.10

# ตั้งค่าตัวแปร environment ที่จำเป็น
ENV AIRFLOW_HOME=/opt/airflow

# ติดตั้ง dependencies เบื้องต้น
USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    default-libmysqlclient-dev \
    && apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# เปลี่ยนไปใช้ user airflow (ไม่ใช่ root)
USER airflow

# ติดตั้ง Python packages ที่ต้องการ
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt


# เพิ่ม DAGs ไฟล์และ Plugin ไฟล์ (ถ้ามี)
COPY ./dags $AIRFLOW_HOME/dags
# COPY ./plugins $AIRFLOW_HOME/plugins

# ใช้ entrypoint ของ Airflow
# ENTRYPOINT ["docker-entrypoint.sh"]
# # ENTRYPOINT ["entrypoint"]
# # ENTRYPOINT ["/usr/bin/dumb-init", "--"]
# CMD ["airflow", "webserver"]
# CMD ["webserver"]
# CMD ["bash", "-c", "airflow db upgrade && airflow webserver"]

