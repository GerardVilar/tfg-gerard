#!/bin/bash

echo "Waiting for Kafka to accept connections on kafka:9092..."

while ! nc -z kafka 9092; do
  echo "Kafka not available yet..."
  sleep 2
done

echo "Kafka is ready. Starting service..."
exec "$@"