import json
import os
import numpy as np
from kafka import KafkaProducer, KafkaConsumer

# Create a Kafka producer to send messages
def create_producer(bootstrap_servers: str):
    return KafkaProducer(
        bootstrap_servers=bootstrap_servers,
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
    )

# Create a Kafka consumer to receive messages from a specific topic
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

    # Honeypot blending coefficients
    alpha = float(os.environ.get("HP_ALPHA", "0.33"))
    beta  = float(os.environ.get("HP_BETA", "0.67"))

    print(f"[HONEYPOT] Booting honeypot | alpha={alpha} beta={beta}")

    # Consumer to receive the real global model broadcasted by the server
    consumer_real = create_consumer(
        bootstrap, "server.broadcast.real", "honeypot-real"
    )

    # Consumer to receive malicious agents' models
    consumer_mal = create_consumer(
        bootstrap, "server.to.honeypot", "honeypot-mal"
    )

    # Producer to send fake models back to the server
    producer = create_producer(bootstrap)

    # Cache of the last real global model per round
    last_real_by_round = {}  # round -> weights

    # Infinite honeypot processing loop
    while True:
        # Try to read the latest real model (non-blocking)
        msg_real = next(consumer_real, None)
        if msg_real is not None:
            data = msg_real.value
            r = data.get("round")
            w = data.get("weights")

            # Store the real model associated with the round
            if r is not None and w is not None:
                last_real_by_round[r] = w

        # Wait for a malicious model sent by the server
        msg = next(consumer_mal)
        data = msg.value
        round_ = data.get("round")
        agent_id = data.get("agent_id")
        mal_w = data.get("weights")

        # Retrieve the real model for the same or previous round
        real_w = last_real_by_round.get(
            round_,
            last_real_by_round.get(round_ - 1)
        )

        # Fallback in case no real model is available
        if real_w is None:
            real_w = mal_w

        # Convert weight lists to NumPy arrays
        real_arr = np.array(real_w, dtype=np.float32)
        mal_arr  = np.array(mal_w, dtype=np.float32)

        # Blend real and malicious models to generate a fake response
        fake = (alpha * real_arr + beta * mal_arr).tolist()

        # Build response message for the server
        msg_back = {
            "round": round_,
            "agent_id": agent_id,
            "weights": fake
        }

        # Send the fake model back to the server
        producer.send("honeypot.to.server", msg_back)
        producer.flush()
        print(f"[HONEYPOT] Sent fake model for {agent_id} round {round_}")

if __name__ == "__main__":
    main()