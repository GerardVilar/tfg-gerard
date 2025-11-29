import json
import os
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

    print("[HONEYPOT] Booting honeypot")

    consumer = create_consumer(bootstrap, "server.to.honeypot", "honeypot")
    producer = create_producer(bootstrap)

    for msg in consumer:
        data = msg.value
        round_ = data.get("round")
        agent_id = data.get("agent_id")
        weights = data.get("weights")

        print(f"[HONEYPOT] Modelo recibido de {agent_id} en ronda {round_}")

        # Just sends the same model
        fake_weights = weights

        # Respond the server
        msg_back = {
            "round": round_,
            "agent_id": agent_id,
            "weights": fake_weights
        }

        producer.send("honeypot.to.server", msg_back)
        producer.flush()
        print(f"[HONEYPOT] Modelo procesado para {agent_id} enviado al servidor")

if __name__ == "__main__":
    main()
