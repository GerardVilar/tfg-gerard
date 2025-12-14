# server/main.py
import json
import os
import time
import csv
import numpy as np

from kafka import KafkaProducer, KafkaConsumer
from kafka.errors import NoBrokersAvailable
from shared.model_utils import (
    init_model,
    average_models,
    evaluate_vector_on_test,
)

def preview(v):
    return f"{v[:3]} ... ({len(v)} valores)"

def classify_by_distance(agent_models, factor=1.2):
    ids = list(agent_models.keys())
    vecs = {aid: np.array(agent_models[aid], dtype=float) for aid in ids}

    # Average distance
    avg_dist = {}
    for i in ids:
        dists = []
        for j in ids:
            if i == j:
                continue
            d = np.linalg.norm(vecs[i] - vecs[j])
            dists.append(d)
        avg_dist[i] = np.mean(dists)

    # DEBUG: see distances
    print("[SERVER] Medium distances per agent:")
    for aid in ids:
        print(f"   {aid}: {avg_dist[aid]:.4f}")

    # Order (desc)
    sorted_ids = sorted(ids, key=lambda aid: avg_dist[aid], reverse=True)

    if len(sorted_ids) < 3:
        return ids, []

    worst = sorted_ids[0]
    second = sorted_ids[1]

    if avg_dist[worst] > factor * avg_dist[second]:
        malicious = [worst]
        benign = [a for a in ids if a != worst]
    else:
        malicious = []
        benign = ids

    return benign, malicious

def create_producer(bootstrap_servers: str, max_retries: int = 10, delay: float = 3.0):
    for attempt in range(1, max_retries + 1):
        try:
            print(f"[SERVER] Creating KafkaProducer (attempt {attempt}/{max_retries})...")
            producer = KafkaProducer(
                bootstrap_servers=bootstrap_servers,
                api_version=(0, 10, 2),
                value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            )
            print("[SERVER] KafkaProducer created successfully.")
            return producer
        except NoBrokersAvailable:
            print(f"[SERVER] No brokers available yet, retrying in {delay}s...")
            time.sleep(delay)

    raise RuntimeError(
        f"[SERVER] Could not connect to Kafka after {max_retries} attempts."
    )


def create_consumer(bootstrap_servers: str, topic: str, group_id: str):
    return KafkaConsumer(
        topic,
        bootstrap_servers=bootstrap_servers,
        api_version=(0, 10, 2),
        group_id=group_id,
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        auto_offset_reset="earliest",
        enable_auto_commit=True,
    )


def main():
    bootstrap = os.environ.get("KAFKA_BOOTSTRAP_SERVERS", "kafka:9092")

    agents_env = os.environ.get("AGENTS","benign_1,benign_2,benign_3,benign_4,malicious_1")

    agents = [a.strip() for a in agents_env.split(",") if a.strip()]

    num_rounds = int(os.environ.get("NUM_ROUNDS", "10000"))
    model_dim = int(os.environ.get("MODEL_DIM", "4"))

    print(f"[SERVER] Using bootstrap servers: {bootstrap}")
    print(f"[SERVER] Starting with agents: {agents}")
    print(f"[SERVER] Rounds: {num_rounds}, model dim: {model_dim}")

    producer = create_producer(bootstrap)
    consumer_updates = create_consumer(
        bootstrap, "agent.to.server", group_id="server-updates"
    )
    consumer_hp = create_consumer(
        bootstrap, "honeypot.to.server", group_id="server-hp"
    )

    #LOG
    log_path = os.environ.get(
        "SERVER_METRICS_FILE",
        "/app/shared/logs/training_metrics.csv"
    )
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_file = open(log_path, "w", newline="")
    csv_writer = csv.writer(log_file)
    csv_writer.writerow(
        ["round", "num_benign", "num_malicious", "acc_benign", "acc_all"]
    )
    print(f"[SERVER] Logging metrics to {log_path}")

    # Reference model on the system
    global_real_model = init_model(model_dim)
    print("[SERVER] Global model initialized.")

    print("[SERVER] Evaluating initial model on test set...")

    loss, acc = evaluate_vector_on_test(global_real_model, max_samples=5000)
    print(f"[SERVER] Global test accuracy (benign-only): {acc:.4f}")

    # State per agent
    per_agent_model = {a: list(global_real_model) for a in agents}

    current_round = 0

    while current_round < num_rounds:
        current_round += 1
        print(f"\n[SERVER] ==== RONDA {current_round} ====")

        # 1) Send the model to every agent
        for agent_id in agents:
            msg = {
                "round": current_round,
                "agent_id": agent_id,
                "weights": per_agent_model[agent_id],
            }
            producer.send("server.to.agent", msg)
            print(f"[SERVER] -> Sent model to {agent_id}")
        producer.flush()

        # 2) Receive models from every agent
        received_models = {}
        print("[SERVER] Waiting for the agents model...")
        while len(received_models) < len(agents):
            msg = next(consumer_updates)
            data = msg.value
            r = data.get("round")
            agent_id = data.get("agent_id")
            weights = data.get("weights")

            if r != current_round:
                # Old model message, ignore
                continue

            if agent_id not in agents:
                continue

            if agent_id in received_models:
                # Agent on possesion
                continue

            received_models[agent_id] = weights
            print(f"[SERVER] <- Recibido modelo de {agent_id}")

        # 3) Detect of malicious agents
        benign, malicious = classify_by_distance(received_models, factor=1.2)
        
        print(f"[SERVER] Agentes benignos: {benign}")
        print(f"[SERVER] Agentes maliciosos: {malicious}")

        benign_models = [received_models[a] for a in benign] if benign else []
        all_models = list(received_models.values())

        if benign_models:
            benign_avg = average_models(benign_models)
            _, acc_benign = evaluate_vector_on_test(benign_avg, max_samples=5000)
        else:
            acc_benign = None

        all_avg = average_models(all_models)
        _, acc_all = evaluate_vector_on_test(all_avg, max_samples=5000)

        print(f"[SERVER] Accuracy con solo benignos: {acc_benign}")
        print(f"[SERVER] Accuracy con benignos+malicioso: {acc_all}")

        #Log to CSV
        csv_writer.writerow([
            current_round,
            len(benign),
            len(malicious),
            "" if acc_benign is None else float(acc_benign),
            "" if acc_all is None else float(acc_all),
        ])
        log_file.flush()

        # 4) Calculate real model
        if benign:
            benign_models = [received_models[a] for a in benign]
            benign_avg = average_models(benign_models)

            alpha = 0.3
            global_real_model = [
                (1 - alpha) * old + alpha * new
                for old, new in zip(global_real_model, benign_avg)
            ]
        else:
            # Extreme case: Every agent is malicious
            print("[SERVER] Every agent is malicious, using every agent value...")
            global_real_model = average_models(
                list(received_models.values())
            )

        print(f"[SERVER] New REAL model: {preview(global_real_model)}")

        # 5) Notify honeypot the real model
        msg_real = {
            "round": current_round,
            "weights": global_real_model,
        }
        producer.send("server.broadcast.real", msg_real)
        producer.flush()
        print("[SERVER] -> Real model sent to honeypot")

        # 6) If there are malicious agents, send them to the honeypot and wait
        hp_models = {}
        if malicious:
            # Send malicious agents to honeypot
            for mal in malicious:
                msg_mal = {
                    "round": current_round,
                    "agent_id": mal,
                    "weights": received_models[mal],
                }
                producer.send("server.to.honeypot", msg_mal)
                print(f"[SERVER] -> Modelo malicioso enviado al honeypot ({mal})")
            producer.flush()

            # Wait for honeypot responses
            pending_hp = set(malicious)
            print("[SERVER] Esperando modelos honeypot...")
            while pending_hp:
                msg_hp = next(consumer_hp)
                data_hp = msg_hp.value
                r_hp = data_hp.get("round")
                agent_hp = data_hp.get("agent_id")
                weights_hp = data_hp.get("weights")

                if r_hp != current_round:
                    continue
                if agent_hp not in pending_hp:
                    continue

                hp_models[agent_hp] = weights_hp
                pending_hp.remove(agent_hp)
                print(f"[SERVER] <- Modelo honeypot recibido para {agent_hp}")

        # 7) Update the models to be seen for the agents
        for a in agents:
            if a in malicious and a in hp_models:
                per_agent_model[a] = hp_models[a]
            else:
                per_agent_model[a] = global_real_model

        print("[SERVER] Estado per-agent para próxima ronda:")
        for a in agents:
            print(f"   {a}: {preview(per_agent_model[a])}")

        # Pause
        time.sleep(1.0)

    print("[SERVER] Fin de las rondas. Saliendo...")


if __name__ == "__main__":
    main()
