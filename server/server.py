# server/server.py
import os, json, time
import numpy as np
from confluent_kafka import Producer, Consumer
from common.serdes import b64_to_ndarray, ndarray_to_b64

BROKER = os.getenv("KAFKA_BROKER", "kafka:9092")
TOPIC_UPDATES = os.getenv("TOPIC_UPDATES", "fl.model.updates")
TOPIC_GLOBAL = os.getenv("TOPIC_GLOBAL", "fl.model.global")
AGENTS_EXPECTED = [x.strip() for x in os.getenv("AGENTS_EXPECTED", "a0,a1").split(",")]
ROUND_TIMEOUT_SEC = int(os.getenv("ROUND_TIMEOUT_SEC", "15"))

def p() -> Producer:
    return Producer({"bootstrap.servers": BROKER})

def c(group) -> Consumer:
    return Consumer({
        "bootstrap.servers": BROKER,
        "group.id": group,
        "auto.offset.reset": "earliest"
    })

def collect_round(round_id: int):
    cons = c(group="server-updates")
    cons.subscribe([TOPIC_UPDATES])
    deadline = time.time() + ROUND_TIMEOUT_SEC
    updates = {}
    while time.time() < deadline and len(updates) < len(AGENTS_EXPECTED):
        msg = cons.poll(1.0)
        if not msg or msg.error():
            continue
        try:
            payload = json.loads(msg.value())
            if payload.get("type") != "model_update":
                continue
            if payload.get("round") != round_id:
                continue
            aid = payload.get("agent_id")
            if aid in AGENTS_EXPECTED and aid not in updates:
                updates[aid] = b64_to_ndarray(payload["weights"])
                print(f"[server] Round {round_id}: got update from {aid}")
        except Exception:
            pass
    return updates

def aggregate(updates: dict[str, np.ndarray]) -> np.ndarray | None:
    if not updates:
        return None
    stacks = np.stack(list(updates.values()), axis=0)
    return stacks.mean(axis=0).astype(np.float32)

def publish_global(round_id: int, weights: np.ndarray):
    msg = {
        "type": "global_model",
        "round": round_id,
        "weights": ndarray_to_b64(weights),
        "timestamp": time.time(),
        "agents_included": list(AGENTS_EXPECTED),
    }
    pr = p()
    pr.produce(TOPIC_GLOBAL, json.dumps(msg).encode("utf-8"))
    pr.flush()
    print(f"[server] Round {round_id}: global published.")

def main():
    print(f"[server] expecting agents: {AGENTS_EXPECTED}")
    current_global = np.zeros(100, dtype=np.float32)  # estado global inicial (mock)
    for round_id in range(1, 4):
        print(f"[server] Round {round_id}: collecting updates ...")
        updates = collect_round(round_id)
        if not updates:
            print(f"[server] Round {round_id}: no updates received.")
            continue
        # (Aquí después añadirás: distancias, DBSCAN, seguridad, etc.)
        global_weights = aggregate(updates)
        if global_weights is not None:
            publish_global(round_id, global_weights)
            current_global = global_weights

if __name__ == "__main__":
    main()
