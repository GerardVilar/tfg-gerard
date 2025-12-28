# agents/malicious_agent.py
import json
import os
import random
import time
import csv


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

MAX_LOCAL_SAMPLES = 1000

_x_train = _x_train[:MAX_LOCAL_SAMPLES].astype("float32") / 255.0
_y_train = _y_train[:MAX_LOCAL_SAMPLES]
_y_train = tf.keras.utils.to_categorical(_y_train, NUM_CLASSES)

AGENT_ID = os.environ.get("AGENT_ID", "malicious_1")

def get_local_data():
    total = _x_train.shape[0]
    num_shards = 3
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

def local_poison_train(weights_vector, round_num: int, eval_every: int):
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

    train_acc_poison = float(history.history["accuracy"][-1])

    eval_acc_poison = None
    eval_acc_clean = None

    if eval_every > 0 and (round_num % eval_every == 0):
        _, eval_acc_poison_val = model.evaluate(x_local, y_poison, verbose=0)
        _, eval_acc_clean_val = model.evaluate(x_local, y_local_clean, verbose=0)
        eval_acc_poison = float(eval_acc_poison_val)
        eval_acc_clean = float(eval_acc_clean_val)

    return model_to_vector(model), train_acc_poison, eval_acc_poison, eval_acc_clean

def preview(v):
    return f"{v[:3]} ... ({len(v)} values)"

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

    log_path = os.environ.get(
        "AGENT_METRICS_FILE",
        f"/app/shared/logs/{agent_id}_metrics.csv"
    )
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    eval_every = int(os.environ.get("AGENT_EVAL_EVERY", "1"))

    log_file = open(log_path, "w", newline="")
    csv_writer = csv.writer(log_file)
    csv_writer.writerow(["round", "train_acc_poison", "eval_acc_poison", "eval_acc_clean"])
    log_file.flush()

    print(f"[AGENT {agent_id}] Logging local metrics to {log_path} (eval every {eval_every})")

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
        new_weights, train_acc_poison, eval_acc_poison, eval_acc_clean = local_poison_train(weights, r, eval_every)

        print(f"[AGENT {agent_id}] Poison train acc (poison labels): {train_acc_poison:.4f}")
        if eval_acc_poison is not None:
            print(f"[AGENT {agent_id}] Eval acc (poison labels): {eval_acc_poison:.4f}")
            print(f"[AGENT {agent_id}] Eval acc (clean labels): {eval_acc_clean:.4f}")
        else:
            print(f"[AGENT {agent_id}] Local eval skipped (eval_every={eval_every})")

        csv_writer.writerow([
            r,
            train_acc_poison,
            "" if eval_acc_poison is None else eval_acc_poison,
            "" if eval_acc_clean is None else eval_acc_clean,
        ])
        log_file.flush()

        msg_out = {
            "round": r,
            "agent_id": agent_id,
            "weights": new_weights,
        }
        producer.send("agent.to.server", msg_out)
        producer.flush()
        print(f"[AGENT {agent_id}] -> Malicious model sent to the server")

        time.sleep(0)


if __name__ == "__main__":
    main()
