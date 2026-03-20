import pandas as pd
import os

def load_labels(label_file):
    labels = {}
    with open(label_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                labels[parts[0]] = parts[1]
    return labels


def load_tweets(tweet_file):
    tweets = {}
    with open(tweet_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) >= 2:
                tweets[parts[0]] = parts[1]
    return tweets


def build_stub_graph(text):
    return {
        "nodes": [
            {"id": 0, "text": text},
            {"id": 1, "text": f"User reaction: {text[:50]}"},
            {"id": 2, "text": f"Another comment: {text[:50]}"},
        ],
        "edges": [
            {"src": 0, "dst": 1, "weight": 1.0},
            {"src": 0, "dst": 2, "weight": 1.0},
        ]
    }


def process_dataset(folder_path):
    label_path = os.path.join(folder_path, "label.txt")
    tweet_path = os.path.join(folder_path, "source_tweets.txt")

    labels = load_labels(label_path)
    tweets = load_tweets(tweet_path)

    rows = []

    for tweet_id, text in tweets.items():
        label_raw = labels.get(tweet_id, "non-rumor")

        # convert label → binary
        label = 1 if "rumor" in label_raw.lower() else 0

        graph = build_stub_graph(text)

        rows.append({
            "text": text,
            "label": label,
            "graph": graph,
            "image_path": None
        })

    return rows


# ── MAIN ─────────────────────────────

twitter15_path = "data/twitter_raw/twitter15"
twitter16_path = "data/twitter_raw/twitter16"

rows = []
rows += process_dataset(twitter15_path)
rows += process_dataset(twitter16_path)

df = pd.DataFrame(rows)
df.to_csv("data/twitter_misinfo.csv", index=False)

print(f"✅ Created dataset with {len(df)} samples")