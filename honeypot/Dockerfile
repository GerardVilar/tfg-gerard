# honeypot/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Instala netcat y cualquier otro paquete necesario
RUN apt-get update && apt-get install -y netcat-openbsd && rm -rf /var/lib/apt/lists/*

COPY main.py wait-for-kafka.sh ./

RUN chmod +x wait-for-kafka.sh

# Instala confluent-kafka y kafka-python
RUN pip install --no-cache-dir confluent-kafka kafka-python

CMD ["./wait-for-kafka.sh"]
