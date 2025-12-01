# agents/malicious_agent.py
import json
import os
import random
import time

import numpy as np
from kafka import KafkaProducer, KafkaConsumer
from tensorflow.keras.datasets import cifar10

IMG_SIZE = 32 * 32 * 3
NOISE_SCALE = 10

(_x_train, _y_train), _ = cifar10.load_data()

def preview(v):
    return f"{v[:3]} ... ({len(v)} valores)"

def create_producer(bootstrap_servers: str):
    return KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )


def create_consumer(bootstrap_servers: str, topic: str, group_id: str):
    return KafkaConsumer(
        topic,
        bootstrap_servers=bootstrap_servers,
        group_id=group_id,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        auto_offset_reset="earliest",
        enable_auto_commit=True,
    )


def get_random_cifar_vector():
    idx = random.randrange(len(_x_train))
    img = _x_train[idx].astype("float32")  # (32, 32, 3)
    vec = img.flatten()
    return vec.tolist()


def ensure_cifar_vector(weights):
    if not isinstance(weights, list) or len(weights) != IMG_SIZE:
        print("[AGENT] WARNING: invalid weights received, ignoring image.")
        return [0.0] * IMG_SIZE

    return weights


def local_poison_train(weights):
    w = np.array(weights, dtype="float32")
    noise = np.random.normal(loc=0.0, scale=NOISE_SCALE, size=w.shape)  # strong noise
    w = np.clip(w + noise, 0.0, 255.0)
    return w.tolist()


def main():
    bootstrap = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
    agent_id = os.environ.get("AGENT_ID", "malicious_1")

    print(f"[AGENT {agent_id}] Booting MALICIOUS agent (CIFAR-10).")

    producer = create_producer(bootstrap)
    consumer = create_consumer(bootstrap, "server.to.agent", group_id=f"agent-{agent_id}")

    current_round = 0

    for msg in consumer:
        data = msg.value
        r = data.get("round")
        target_agent = data.get("agent_id")

        if target_agent != agent_id:
            continue
        if r <= current_round:
            continue

        current_round = r
        weights = data.get("weights")
        weights = ensure_cifar_vector(weights)

        def preview(v):
            return f"{v[:3]} ... ({len(v)} values)"

        print(f"[AGENT {agent_id}] Round {r}, model received: {preview(weights)}")

        # Malicious training
        new_weights = local_poison_train(weights)
        print(f"[AGENT {agent_id}] Round {r}, trained model: {preview(new_weights)}")

        msg_out = {
            "round": r,
            "agent_id": agent_id,
            "weights": new_weights,
        }
        producer.send("agent.to.server", msg_out)
        producer.flush()
        print(f"[AGENT {agent_id}] -> Malicious model sent to the server")

        time.sleep(0.5)


if __name__ == "__main__":
    main()
