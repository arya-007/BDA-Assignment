🎧 **Multimodal Anxiety Prediction — Real-Time Streaming Pipeline**

A complete real-time anxiety prediction system built using Kafka, Python, ML models, and Streamlit, integrating audio, text, and video features from the DAIC-WOZ dataset.
<br>
📥 **0. Download Dataset (DAIC-WOZ)**

This project uses the DAIC-WOZ Depression Dataset.

Download the participant folders manually:

🔗 https://dcapswoz.ict.usc.edu/wwwdaicwoz/

You will receive folders like:

```bash
300_P/
301_P/
302_P/
...

```

Place all downloaded folders here:

```bash
data/raw/
```

📁 Folder Structure
```bash
data/
 └── raw/
      ├── 300_P/
      ├── 301_P/
      ├── 302_P/
      └── ...
```

<br>

🚀 **1. Environment Setup**

Install Homebrew (macOS)
```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Install Necessary Tools
```bash
brew install python
brew install kafka
```

Install Python Dependencies
```bash
pip install -r requirements.txt
```


<br>

⚡ **2. Kafka Setup (KRaft Mode)**
Start Kafka
```bash
brew services start kafka
```

Verify Kafka Install Location
```bash
brew --prefix kafka
```

Expected:
```bash
/opt/homebrew/opt/kafka
```

Add Kafka CLI Tools to PATH
```bash
echo 'export PATH="$(brew --prefix kafka)/libexec/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

<br>

📌 **3. Create Kafka Topics**

Run:
```bash
kafka-topics.sh --bootstrap-server localhost:9092 --create --topic audio_features
kafka-topics.sh --bootstrap-server localhost:9092 --create --topic text_features
kafka-topics.sh --bootstrap-server localhost:9092 --create --topic video_features
kafka-topics.sh --bootstrap-server localhost:9092 --create --topic anxiety_predictions
```

List all topics:
```bash
kafka-topics.sh --list --bootstrap-server localhost:9092
```

<br>

🎤 **4. Preprocessing Pipeline**
Audio Processing
```bash
python3 src/audio_processing.py --participant 300_P
```

Text Processing
```bash
python3 src/text_processing.py --participant 300_P
```

Video Processing
```bash
python3 src/video_processing.py --participant 300_P
```

Feature Fusion
```bash
python3 src/feature_fusion.py
```

Output:
```bash
data/processed/fused/all_features.parquet
```

<br>

🧠 **5. Train ML Model**

Uses:

fused features

train_split_depression.csv

Run:
```bash
python3 src/model_training.py
```

Model saved at:
```bash
data/processed/fused/model.joblib
```

<br>

🔄 **6. Real-Time Streaming Pipeline**
Start Kafka Producer (sends features)
```bash
python3 src/kafka_producer.py
```

Start Stream Consumer (makes predictions)
```bash
python3 src/stream_consumer.py
```

Outputs parquet files to:
```bash
data/stream_output/predictions/
```

<br>

📁 **7. Convert Stream Output (Parquet → CSV)**

After the stream completes:
```bash
python3 merge_parquets_to_csv.py
```

This generates:
```bash
data/stream_output/predictions/predictions.csv
```

<br>

🖥️ **8. Streamlit App**
Install Streamlit
```bash
pip3 install streamlit
```

Ensure Streamlit is in PATH (macOS specific)
```bash
echo 'export PATH=$PATH:/Users/$USER/Library/Python/3.9/bin' >> ~/.zshrc
source ~/.zshrc
```

Run Streamlit App
```bash
streamlit run app.py
```

App Features

Loads predictions.csv
Shows first 10 columns in an interactive table