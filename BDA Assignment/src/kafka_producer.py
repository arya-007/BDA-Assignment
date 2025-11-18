# src/kafka_producer.py
import json
import time
import os
import pandas as pd
from kafka import KafkaProducer
from utils import list_participants

BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP", "localhost:9092")
producer = KafkaProducer(bootstrap_servers=BOOTSTRAP,
                         value_serializer=lambda v: json.dumps(v).encode('utf-8'))

def send_topic(topic, payload):
    producer.send(topic, payload)
    producer.flush()

def produce_all_once(raw_dir="data/raw", processed_dir="data/processed", wait=0.1):
    participants = list_participants(raw_dir)
    for pid in participants:
        print("Producing features for", pid)
        # look for processed CSVs; you may produce per-participant fused features,
        # or produce modality-specific messages.
        # We'll produce 3 messages: audio, text, video (if they exist)
        audio_p = os.path.join(processed_dir, "audio_features", f"{pid}.csv")
        audio_ext = os.path.join(processed_dir, "audio_features_ext", f"{pid}.csv")
        text_p = os.path.join(processed_dir, "text_features", f"{pid}.csv")
        video_p = os.path.join(processed_dir, "video_features", f"{pid}.csv")

        # helper to load csv to dict
        def load_dict(path):
            if os.path.exists(path):
                d = pd.read_csv(path).to_dict(orient="records")
                return d[0] if d else {}
            return {}

        audio = load_dict(audio_p)
        if audio_ext:
            audio_ext_d = load_dict(audio_ext)
            audio.update(audio_ext_d or {})

        text = load_dict(text_p)
        video = load_dict(video_p)

        timestamp = time.time()
        if audio:
            send_topic("audio_features", {"participant_id": pid, "timestamp": timestamp, "features": audio})
        if text:
            send_topic("text_features", {"participant_id": pid, "timestamp": timestamp, "features": text})
        if video:
            send_topic("video_features", {"participant_id": pid, "timestamp": timestamp, "features": video})

        time.sleep(wait)

if __name__ == "__main__":
    produce_all_once()
    print("Done producing.")
