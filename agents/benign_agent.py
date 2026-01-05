# agents/benign_agent.py
import json
import os
import random
import time
import csv


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from shared.model_utils import (
    build_model,
    vector_to_model,
    model_to_vector,
)
from tensorflow.keras.datasets import cifar10
import tensorflow as tf

import numpy as np
from kafka import KafkaProducer, KafkaConsumer

IMG_SHAPE = (32, 32, 3)
NUM_CLASSES = 10

IMG_SIZE = 32 * 32 * 3

(_x_train, _y_train), _ = cifar10.load_data()

MAX_LOCAL_SAMPLES = 1000

_x_train = _x_train[:MAX_LOCAL_SAMPLES].astype("float32") / 255.0
_y_train = _y_train[:MAX_LOCAL_SAMPLES]
_y_train = tf.keras.utils.to_categorical(_y_train, NUM_CLASSES)

AGENT_ID = os.environ.get("AGENT_ID", "benign_1")

_rng = random.Random(hash(AGENT_ID) & 0xFFFFFFFF)
NOISE_SCALE = _rng.uniform(0.5, 2)

# Every agent has one different
def get_local_data():
    total = _x_train.shape[0]
    num_shards = 4

    try:
        suffix = int(AGENT_ID.split("_")[-1])
        idx = (suffix - 1) % num_shards
    except ValueError:
        idx = 0

    shard_size = total // num_shards
    start = idx * shard_size
    end = (idx + 1) * shard_size

    print(f"[AGENT {AGENT_ID}] Using shard {idx} [{start}:{end}] of {total}")
    return _x_train[start:end], _y_train[start:end]


x_local, y_local = get_local_data()


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


def get_random_cifar_vector():
    idx = random.randrange(len(_x_train))
    img = _x_train[idx].astype("float32")  # (32, 32, 3)
    vec = img.flatten()  # (3072,)
    return vec.tolist()


def ensure_cifar_vector(weights):
    if not isinstance(weights, list) or len(weights) != IMG_SIZE:
        print("[AGENT] WARNING: invalid weights received, ignoring image.")
        return [0.0] * IMG_SIZE

    return weights


def local_train(weights_vector, round_num: int, eval_every: int):
    # Rebuild model
    model = build_model()
    vector_to_model(model, weights_vector)

    # Train
    history = model.fit(
        x_local,
        y_local,
        epochs=1,
        batch_size=64,
        verbose=0,
    )

    train_acc = float(history.history["accuracy"][-1])
    
    eval_acc = None
    if eval_every > 0 and (round_num % eval_every == 0):
        _, eval_acc_val = model.evaluate(x_local, y_local, verbose=0)
        eval_acc = float(eval_acc_val)

    return model_to_vector(model), train_acc, eval_acc


def main():
    bootstrap = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")

    print(f"[AGENT {AGENT_ID}] Booting BENIGN agent (CIFAR-10).")

    producer = create_producer(bootstrap)
    consumer = create_consumer(
        bootstrap, "server.to.agent", group_id=f"agent-{AGENT_ID}"
    )

    log_path = os.environ.get(
        "AGENT_METRICS_FILE",
        f"/app/shared/logs/{AGENT_ID}_metrics.csv"
    )
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    
    eval_every = int(os.environ.get("AGENT_EVAL_EVERY", "1"))

    log_file = open(log_path, "w", newline="")
    csv_writer = csv.writer(log_file)
    csv_writer.writerow(["round", "train_acc", "eval_acc_clean"])
    log_file.flush()

    current_round = 0

    for msg in consumer:
        data = msg.value
        r = data.get("round")
        target_agent = data.get("agent_id")

        if target_agent != AGENT_ID:
            # Message for other agent
            continue
        if r <= current_round:
            # Old round
            continue

        current_round = r
        weights = data.get("weights")

        print(f"[AGENT {AGENT_ID}] Round {r}, model received: {preview(weights)}")

        # Train locally
        new_weights, train_acc, eval_acc = local_train(weights, r, eval_every)

        print(f"[AGENT {AGENT_ID}] Local train accuracy: {train_acc:.4f}")
        if eval_acc is not None:
            print(f"[AGENT {AGENT_ID}] Local eval accuracy (clean): {eval_acc:.4f}")
        else:
            print(f"[AGENT {AGENT_ID}] Local eval skipped (eval_every={eval_every})")

        csv_writer.writerow([
            r,
            train_acc,
            "" if eval_acc is None else eval_acc,
        ])
        log_file.flush()

        # Send to server
        msg_out = {
            "round": r,
            "agent_id": AGENT_ID,
            "weights": new_weights,
        }
        producer.send("agent.to.server", msg_out)
        producer.flush()
        print(f"[AGENT {AGENT_ID}] -> Model sent to server")

        time.sleep(0)


if __name__ == "__main__":
    main()
