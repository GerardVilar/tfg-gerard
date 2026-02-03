# server/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Installs necessary packages
RUN apt-get update && apt-get install -y netcat-openbsd && rm -rf /var/lib/apt/lists/*

COPY main.py wait-for-kafka.sh ./

RUN chmod +x wait-for-kafka.sh

# Installs confluent-kafka and kafka-python
RUN pip install --no-cache-dir confluent-kafka kafka-python numpy Pillow tensorflow

CMD ["./wait-for-kafka.sh"]