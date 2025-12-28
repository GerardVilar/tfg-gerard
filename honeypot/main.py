import json
import os
import numpy as np
from kafka import KafkaProducer, KafkaConsumer

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
    alpha = float(os.environ.get("HP_ALPHA", "0.33"))
    beta  = float(os.environ.get("HP_BETA", "0.67"))

    print(f"[HONEYPOT] Booting honeypot | alpha={alpha} beta={beta}")

    consumer_real = create_consumer(bootstrap, "server.broadcast.real", "honeypot-real")
    
    consumer_mal = create_consumer(bootstrap, "server.to.honeypot", "honeypot-mal")
    producer = create_producer(bootstrap)

    last_real_by_round = {}  # round -> weights

    while True:
        msg_real = next(consumer_real, None)
        if msg_real is not None:
            data = msg_real.value
            r = data.get("round")
            w = data.get("weights")
            if r is not None and w is not None:
                last_real_by_round[r] = w

        msg = next(consumer_mal)
        data = msg.value
        round_ = data.get("round")
        agent_id = data.get("agent_id")
        mal_w = data.get("weights")

        real_w = last_real_by_round.get(round_, last_real_by_round.get(round_-1))
        if real_w is None:
            real_w = mal_w  # fallback

        real_arr = np.array(real_w, dtype=np.float32)
        mal_arr  = np.array(mal_w, dtype=np.float32)

        fake = (alpha * real_arr + beta * mal_arr).tolist()

        msg_back = {
            "round": round_,
            "agent_id": agent_id,
            "weights": fake
        }

        producer.send("honeypot.to.server", msg_back)
        producer.flush()
        print(f"[HONEYPOT] Sent fake model for {agent_id} round {round_}")

if __name__ == "__main__":
    main()
