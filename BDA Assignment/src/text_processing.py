# src/text_processing.py
import pandas as pd, numpy as np, os, sys
from transformers import pipeline, AutoTokenizer, AutoModel
import torch
from utils import save_df

def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

def get_text_embedding(text, tokenizer, model, device='cpu'):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
    inputs = {k:v.to(device) for k,v in inputs.items()}
    with torch.no_grad():
        model_output = model(**inputs)
    emb = mean_pooling(model_output, inputs['attention_mask'])
    return emb.cpu().numpy()[0]

def extract_text(transcript_path, tokenizer, model, sentiment_pipeline):
    df = pd.read_csv(transcript_path)
    # find likely transcript column
    text_cols = [c for c in df.columns if any(k in c.lower() for k in ['value','trans','text','utter'])]
    if text_cols:
        text = " ".join(df[text_cols[0]].astype(str).tolist())
    else:
        text = " ".join(df.astype(str).values.flatten().tolist())
    if not text.strip():
        return pd.DataFrame([{"word_count":0}])
    sent = sentiment_pipeline(text[:512])[0]
    emb = get_text_embedding(text[:512], tokenizer, model)
    feat = {"sentiment_label": sent["label"], "sentiment_score": float(sent["score"]), "word_count": len(text.split())}
    for i in range(min(64, len(emb))):
        feat[f"emb_{i}"] = float(emb[i])
    return pd.DataFrame([feat])

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--participant", required=True)
    parser.add_argument("--raw_dir", default="data/raw")
    parser.add_argument("--out_dir", default="data/processed/text_features")
    args = parser.parse_args()

    pid=args.participant
    folder=os.path.join(args.raw_dir,pid)
    # find transcript file
    tfile = os.path.join(folder, f"{pid.split('_')[0]}_TRANSCRIPT.csv")
    if not os.path.exists(tfile):
        tlist=[os.path.join(folder,f) for f in os.listdir(folder) if "transcript" in f.lower()]
        tfile=tlist[0] if tlist else None
    if not tfile:
        print("No transcript for", pid); sys.exit(1)

    # small models for speed
    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    sentiment = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

    df = extract_text(tfile, tokenizer, model, sentiment)
    save_df(df, args.out_dir, pid)
    print("Saved text features for", pid)
