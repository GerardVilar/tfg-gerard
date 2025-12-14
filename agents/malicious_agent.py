# agents/malicious_agent.py
import json
import os
import random
import time

import numpy as np
from kafka import KafkaProducer, KafkaConsumer
from shared.model_utils import (
    build_model,
    vector_to_model,
    model_to_vector,
)
from tensorflow.keras.datasets import cifar10
import tensorflow as tf

IMG_SHAPE = (32, 32, 3)
NUM_CLASSES = 10

IMG_SIZE = 32 * 32 * 3
NOISE_SCALE = 10

(_x_train, _y_train), _ = cifar10.load_data()

MAX_LOCAL_SAMPLES = 5000

_x_train = _x_train[:MAX_LOCAL_SAMPLES].astype("float32") / 255.0
_y_train = _y_train[:MAX_LOCAL_SAMPLES]
_y_train = tf.keras.utils.to_categorical(_y_train, NUM_CLASSES)

AGENT_ID = os.environ.get("AGENT_ID", "malicious_1")

def get_local_data():
    total = _x_train.shape[0]
    num_shards = 5
    idx = num_shards - 1

    shard_size = total // num_shards
    start = idx * shard_size
    end = (idx + 1) * shard_size

    print(f"[AGENT {AGENT_ID}] Using shard {idx} [{start}:{end}] of {total} (MALICIOUS)")
    return _x_train[start:end], _y_train[start:end]

x_local, y_local_clean = get_local_data()

def make_poison_labels(y):
    y_poison = y.copy()
    y_idx = np.argmax(y_poison, axis=1)
    y_idx = (y_idx + 1) % NUM_CLASSES
    y_poison = tf.keras.utils.to_categorical(y_idx, NUM_CLASSES)
    return y_poison

def local_poison_train(weights_vector):
    model = build_model()
    vector_to_model(model, weights_vector)

    y_poison = make_poison_labels(y_local_clean)

    history = model.fit(
        x_local,
        y_poison,
        epochs=3,     # Lower or higher depending on the level of alteration desired
        batch_size=64,
        verbose=0,
    )

    acc = history.history["accuracy"][-1]
    print(f"[AGENT {AGENT_ID}] Poison local train 'accuracy' (on wrong labels): {acc:.4f}")

    return model_to_vector(model)

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
