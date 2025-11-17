# agent/agent.py
import os, time, json, numpy as np
from confluent_kafka import Producer, Consumer
from common.serdes import ndarray_to_b64, b64_to_ndarray

BROKER = os.getenv("KAFKA_BROKER", "kafka:9092")
TOPIC_UPDATES = os.getenv("TOPIC_UPDATES", "fl.model.updates")
TOPIC_GLOBAL = os.getenv("TOPIC_GLOBAL", "fl.model.global")
AGENT_ID = os.getenv("AGENT_ID", "a0")
MODE = os.getenv("MODE", "MOCK")
EPOCHS = int(os.getenv("EPOCHS", "1"))

def p() -> Producer:
    return Producer({"bootstrap.servers": BROKER})

def c(group) -> Consumer:
    return Consumer({
        "bootstrap.servers": BROKER,
        "group.id": group,
        "auto.offset.reset": "earliest"
    })

def train_local(prev_global: np.ndarray | None) -> np.ndarray:
    if prev_global is None:
        model = np.zeros(100, dtype=np.float32)
    else:
        model = prev_global.astype(np.float32).copy()
    for _ in range(EPOCHS):
        model += np.random.normal(0, 0.01, size=model.shape).astype(np.float32)
    return model

def publish_update(round_id: int, weights: np.ndarray):
    msg = {
        "type": "model_update",
        "round": round_id,
        "agent_id": AGENT_ID,
        "weights": ndarray_to_b64(weights),
        "timestamp": time.time(),
    }
    pr = p()
    pr.produce(TOPIC_UPDATES, json.dumps(msg).encode("utf-8"))
    pr.flush()

def listen_global(round_id: int, timeout_sec=15) -> np.ndarray | None:
    cons = c(group=f"{AGENT_ID}-global")
    cons.subscribe([TOPIC_GLOBAL])
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        msg = cons.poll(1.0)
        if not msg or msg.error():
            continue
        try:
            payload = json.loads(msg.value())
            if payload.get("type") == "global_model" and payload.get("round") == round_id:
                return b64_to_ndarray(payload["weights"])
        except Exception:
            pass
    return None

def main():
    print(f"[{AGENT_ID}] starting; mode={MODE}")
    current_global = None
    for round_id in range(1, 4):
        print(f"[{AGENT_ID}] Round {round_id}: training local ...")
        local_weights = train_local(current_global)
        publish_update(round_id, local_weights)
        print(f"[{AGENT_ID}] Round {round_id}: waiting global ...")
        new_global = listen_global(round_id)
        if new_global is not None:
            current_global = new_global
            print(f"[{AGENT_ID}] Round {round_id}: global received.")
        else:
            print(f"[{AGENT_ID}] Round {round_id}: no global received (timeout).")

if __name__ == "__main__":
    main()
