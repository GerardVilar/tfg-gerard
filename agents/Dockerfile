# agents/Dockerfile
FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y netcat-openbsd && rm -rf /var/lib/apt/lists/*

COPY benign_agent.py malicious_agent.py wait-for-kafka.sh ./
RUN chmod +x wait-for-kafka.sh

RUN pip install --no-cache-dir confluent-kafka kafka-python numpy tensorflow

CMD ["./wait-for-kafka.sh", "python", "benign_agent.py"]
