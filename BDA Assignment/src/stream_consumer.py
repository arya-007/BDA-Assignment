# src/stream_consumer.py
"""
Simple Kafka consumer-based streamer that:
- subscribes to audio_features, text_features, video_features
- keeps the latest features per participant per modality
- fuses numeric features
- loads model.joblib (expects keys: model, imputer, scaler, feature_names)
- predicts anxiety severity and writes results to Parquet and optionally to output Kafka topic
"""

import os
import json
import time
import joblib
import signal
import threading
from collections import defaultdict
from kafka import KafkaConsumer, KafkaProducer
import pandas as pd
import numpy as np
from pyarrow import parquet as pq
from pyarrow import Table

# CONFIG
KAFKA_BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP", "localhost:9092")
TOPICS = ["audio_features", "text_features", "video_features"]
OUTPUT_TOPIC = os.environ.get("OUTPUT_TOPIC", "anxiety_predictions")  # optional
PARQUET_OUTDIR = os.environ.get("PARQUET_OUTDIR", "data/stream_output/predictions")
MODEL_PATH = os.environ.get("MODEL_PATH", "data/processed/fused/model.joblib")
WRITE_TO_KAFKA = True  # set to True to publish predictions back to Kafka

# Globals
latest = defaultdict(lambda: {"audio": None, "text": None, "video": None, "last_updated": None})
stop_event = threading.Event()

# Load model
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError("Model not found at: " + MODEL_PATH)
bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
imputer = bundle["imputer"]
scaler = bundle["scaler"]
feature_names = list(bundle["feature_names"])

# prepare Kafka producer if needed
producer = None
if WRITE_TO_KAFKA:
    producer = KafkaProducer(
        bootstrap_servers=KAFKA_BOOTSTRAP,
        value_serializer=lambda v: json.dumps(v).encode("utf-8")
    )

# ensure output dir
os.makedirs(PARQUET_OUTDIR, exist_ok=True)

def flatten_numeric(d):
    """Return dict with numeric values only (floats)."""
    out = {}
    if not isinstance(d, dict):
        return out
    for k, v in d.items():
        try:
            out[str(k)] = float(v)
        except Exception:
            # skip non-numeric
            pass
    return out

def fuse_features(pid):
    """Merge latest modality dicts into one flat dict (numeric values only)."""
    rec = latest[pid]
    merged = {}
    for mod in ("audio", "text", "video"):
        if rec.get(mod):
            merged.update(flatten_numeric(rec[mod].get("features", rec[mod])))
    return merged

def predict_from_flat_dict(flat_dict):
    """Create array in feature_names order, impute, scale, predict."""
    arr = np.array([flat_dict.get(fn, np.nan) for fn in feature_names], dtype=float).reshape(1, -1)
    arr_imp = imputer.transform(arr)
    arr_scaled = scaler.transform(arr_imp)
    pred = model.predict(arr_scaled)[0]
    return float(pred)

def write_prediction_row(pid, fused_features, pred, timestamp=None):
    """Append a parquet file (per time chunk) or write single file. We'll append by creating tiny parquet files."""
    if timestamp is None:
        timestamp = time.time()
    row = {"participant_id": pid, "timestamp": float(timestamp), "anxiety_score": float(pred)}
    # write also selected fused features (you can write entire dict as JSON string)
    # flatten dict to columns with prefix f_
    for k, v in fused_features.items():
        # sanitize column name
        col = f"f_{k}".replace(" ", "_").replace("-", "_")
        row[col] = float(v)
    # convert to DataFrame then to parquet (append by writing new file)
    df = pd.DataFrame([row])
    out_path = os.path.join(PARQUET_OUTDIR, f"pred_{pid}_{int(timestamp)}.parquet")
    table = Table.from_pandas(df)
    pq.write_table(table, out_path)
    return out_path

def on_message(partition, raw_value):
    """
    Called when a new message arrives. raw_value is a dict with keys:
    { "participant_id":..., "timestamp":..., "features": {...} }
    We'll update latest[...] and if all modalities available, predict.
    """
    pid = raw_value.get("participant_id")
    if not pid:
        return
    topic = partition  # passed topic name into this function handler
    now = time.time()
    latest[pid][topic.split("_")[0]] = raw_value  # topic "audio_features" -> key "audio"
    latest[pid]["last_updated"] = now

    # Simple policy: if we have at least one modality, predict using whatever exists.
    fused = fuse_features(pid)
    if not fused:
        return
    try:
        pred = predict_from_flat_dict(fused)
    except Exception as e:
        print("Prediction failed for", pid, "error:", e)
        return

    out_file = write_prediction_row(pid, fused, pred, timestamp=raw_value.get("timestamp", now))
    print(f"[{time.strftime('%H:%M:%S')}] Predicted {pred:.3f} for {pid} -> saved {out_file}")

    # optionally push to Kafka output topic
    if WRITE_TO_KAFKA and producer:
        producer.send(OUTPUT_TOPIC, {"participant_id": pid, "timestamp": now, "anxiety_score": pred})
        producer.flush()

def consumer_loop():
    # Create a consumer subscribing to all topics
    consumer = KafkaConsumer(
        *TOPICS,
        bootstrap_servers=KAFKA_BOOTSTRAP,
        value_deserializer=lambda m: json.loads(m.decode("utf-8")),
        auto_offset_reset="earliest",
        enable_auto_commit=True,
        consumer_timeout_ms=1000  # so .poll() will return after idle
    )
    print("Connected to Kafka. Subscribed to topics:", TOPICS)
    try:
        while not stop_event.is_set():
            for msg in consumer:
                topic = msg.topic  # e.g., "audio_features"
                value = msg.value
                # process
                on_message(topic, value)
                if stop_event.is_set():
                    break
            # loop waits for next poll; consumer will timeout after consumer_timeout_ms
    except Exception as e:
        print("Consumer exception:", e)
    finally:
        consumer.close()

def shutdown(signum, frame):
    print("Shutting down consumer...")
    stop_event.set()

if __name__ == "__main__":
    # handle signals for graceful exit
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    consumer_loop()
