import json
import os
import random
import time
import csv

import numpy as np
from kafka import KafkaProducer, KafkaConsumer

# Utilities for model construction and weight conversion
from shared.model_utils import (
    build_model,
    vector_to_model,
    model_to_vector,
)

# CIFAR-10 dataset utilities
from tensorflow.keras.datasets import cifar10
import tensorflow as tf

# Number of classes in CIFAR-10
NUM_CLASSES = 10

# Attack configuration parameters
ATTACK_MODE = os.environ.get("ATTACK_MODE", "vehicles_to_car")  # shift | vehicles_to_car
TARGET_CLASS = int(os.environ.get("TARGET_CLASS", "1"))         # automobile = 1
VEHICLE_CLASSES = {0, 1, 8, 9}                                  # vehicle-related classes
ATTACK_START_ROUND = int(os.environ.get("ATTACK_START_ROUND", "10"))
AGENT_EVAL_EVERY = int(os.environ.get("AGENT_EVAL_EVERY", "1"))

# Load CIFAR-10 dataset
(_x_train, _y_train), _ = cifar10.load_data()

# Limit number of local samples
MAX_LOCAL_SAMPLES = 1000

# Normalize images and convert labels to categorical format
_x_train = _x_train[:MAX_LOCAL_SAMPLES].astype("float32") / 255.0
_y_train = _y_train[:MAX_LOCAL_SAMPLES]
_y_train = tf.keras.utils.to_categorical(_y_train, NUM_CLASSES)

# Agent identifier
AGENT_ID = os.environ.get("AGENT_ID", "malicious_1")


# Select a fixed shard of data for the malicious agent
def get_local_data():
    total = _x_train.shape[0]
    num_shards = 4

    # Malicious agent always uses the last shard
    idx = num_shards - 1

    shard_size = total // num_shards
    start = idx * shard_size
    end = (idx + 1) * shard_size

    print(f"[AGENT {AGENT_ID}] Using shard {idx} [{start}:{end}] of {total} (MALICIOUS)")
    return _x_train[start:end], _y_train[start:end]


# Load local data for the malicious agent
x_local, y_local_clean = get_local_data()


# Perform benign local training (before attack activation)
def local_benign_train(weights_vector, round_num: int, eval_every: int):
    # Rebuild model architecture
    model = build_model()

    # Load received global weights
    vector_to_model(model, weights_vector)

    # Train normally on clean labels
    history = model.fit(
        x_local,
        y_local_clean,
        epochs=1,
        batch_size=64,
        verbose=0,
    )

    # Extract training accuracy
    train_acc = float(history.history["accuracy"][-1])

    eval_acc = None

    # Optionally evaluate on clean data
    if eval_every > 0 and (round_num % eval_every == 0):
        _, eval_acc_val = model.evaluate(x_local, y_local_clean, verbose=0)
        eval_acc = float(eval_acc_val)

    return model_to_vector(model), train_acc, eval_acc


# Generate poisoned labels according to the selected attack mode
def make_poison_labels(y_clean):
    # Convert one-hot labels to class indices
    y_idx = np.argmax(y_clean, axis=1)

    if ATTACK_MODE == "vehicles_to_car":
        # Map all vehicle classes to the target class
        y_poison_idx = y_idx.copy()
        mask = np.isin(y_idx, list(VEHICLE_CLASSES))
        y_poison_idx[mask] = TARGET_CLASS
        return tf.keras.utils.to_categorical(y_poison_idx, NUM_CLASSES)

    # Default attack: cyclic label shift
    y_poison_idx = (y_idx + 1) % NUM_CLASSES
    return tf.keras.utils.to_categorical(y_poison_idx, NUM_CLASSES)


# Perform malicious local training with poisoned labels
def local_poison_train(weights_vector, round_num: int, eval_every: int):
    # Rebuild model architecture
    model = build_model()

    # Load received global weights
    vector_to_model(model, weights_vector)

    # Generate poisoned labels
    y_train_used = make_poison_labels(y_local_clean)

    # Train model using poisoned labels
    history = model.fit(
        x_local,
        y_train_used,
        epochs=3,      # Increased epochs to amplify the attack effect
        batch_size=64,
        verbose=0,
    )

    print(f"[AGENT {AGENT_ID}] ATTACK_MODE={ATTACK_MODE}, TARGET_CLASS={TARGET_CLASS}")

    # Extract poisoned training accuracy
    train_acc_poison = float(history.history["accuracy"][-1])

    eval_acc_poison = None
    eval_acc_clean = None

    # Optionally evaluate on poisoned and clean labels
    if eval_every > 0 and (round_num % eval_every == 0):
        _, eval_acc_poison_val = model.evaluate(x_local, y_train_used, verbose=0)
        _, eval_acc_clean_val = model.evaluate(x_local, y_local_clean, verbose=0)
        eval_acc_poison = float(eval_acc_poison_val)
        eval_acc_clean = float(eval_acc_clean_val)

    return model_to_vector(model), train_acc_poison, eval_acc_poison, eval_acc_clean


# Utility function to preview a model vector
def preview(v):
    return f"{v[:3]} ... ({len(v)} values)"


# Create a Kafka producer for sending models to the server
def create_producer(bootstrap_servers: str):
    return KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )


# Create a Kafka consumer for receiving models from the server
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
    # Kafka bootstrap server address
    bootstrap = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")
    agent_id = os.environ.get("AGENT_ID", "malicious_1")

    print(f"[AGENT {agent_id}] Booting MALICIOUS agent (CIFAR-10).")

    # Initialize Kafka producer and consumer
    producer = create_producer(bootstrap)
    consumer = create_consumer(
        bootstrap, "server.to.agent", group_id=f"agent-{agent_id}"
    )

    # Setup metrics logging
    log_path = os.environ.get(
        "AGENT_METRICS_FILE",
        f"/app/shared/logs/{agent_id}_metrics.csv"
    )
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    eval_every = int(os.environ.get("AGENT_EVAL_EVERY", "1"))

    log_file = open(log_path, "w", newline="")
    csv_writer = csv.writer(log_file)
    csv_writer.writerow([
        "round",
        "train_acc_poison",
        "eval_acc_poison",
        "eval_acc_clean"
    ])
    log_file.flush()

    print(f"[AGENT {agent_id}] Logging local metrics to {log_path} (eval every {eval_every})")

    current_round = 0

    # Main message-processing loop
    for msg in consumer:
        data = msg.value
        r = data.get("round")
        target_agent = data.get("agent_id")

        # Ignore messages not addressed to this agent
        if target_agent != agent_id:
            continue

        # Ignore outdated rounds
        if r <= current_round:
            continue

        current_round = r
        weights = data.get("weights")

        print(f"[AGENT {agent_id}] Round {r}, model received: {preview(weights)}")

        # Determine whether the attack is active
        attack_active = (r >= ATTACK_START_ROUND)

        if not attack_active:
            # Benign behavior before attack start
            new_weights, train_acc_clean, eval_acc_clean = local_benign_train(
                weights, r, AGENT_EVAL_EVERY
            )
            train_acc_poison = None
            eval_acc_poison = None
            print(f"[AGENT {agent_id}] Benign mode (no attack yet).")
        else:
            # Malicious behavior with poisoned training
            new_weights, train_acc_poison, eval_acc_poison, eval_acc_clean = local_poison_train(
                weights, r, AGENT_EVAL_EVERY
            )
            train_acc_clean = None
            print(f"[AGENT {agent_id}] ATTACK ACTIVE (mode={ATTACK_MODE})")

        # Log accuracy information
        if train_acc_poison is not None:
            print(f"[AGENT {agent_id}] Poison train acc (poison labels): {train_acc_poison:.4f}")
        if eval_acc_poison is not None:
            print(f"[AGENT {agent_id}] Eval acc (poison labels): {eval_acc_poison:.4f}")
            print(f"[AGENT {agent_id}] Eval acc (clean labels): {eval_acc_clean:.4f}")
        else:
            print(f"[AGENT {agent_id}] Local eval skipped (eval_every={eval_every})")

        # Write metrics to CSV
        csv_writer.writerow([
            r,
            train_acc_poison,
            "" if eval_acc_poison is None else eval_acc_poison,
            "" if eval_acc_clean is None else eval_acc_clean,
        ])
        log_file.flush()

        # Send updated model back to the server
        msg_out = {
            "round": r,
            "agent_id": agent_id,
            "weights": new_weights,
        }
        producer.send("agent.to.server", msg_out)
        producer.flush()
        print(f"[AGENT {agent_id}] -> Malicious model sent to the server")

        # Optional delay between rounds
        time.sleep(0)


if __name__ == "__main__":
    main()
