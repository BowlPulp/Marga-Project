"""
========================================================================
MODEL 1: BASELINE MODEL
========================================================================
Architecture  : Single-Agent
Modality      : Text-Only
Retrieval     : None
Graph Context : None
Datasets      : FakeNewsNet, Twitter/Social Graph (text fields only)
Framework     : PyTorch + HuggingFace Transformers
========================================================================

PSEUDOCODE / DESIGN SPEC
--------------------------
INPUT:
    - claim_text (str): The raw news headline or tweet text

PIPELINE:
    1. Tokenise claim_text using BERT tokenizer
    2. Encode via pre-trained BERT (bert-base-uncased)
    3. Pool [CLS] token embedding  →  768-dim vector
    4. Pass through 2-layer MLP classifier
       - Layer 1: Linear(768 → 256) + ReLU + Dropout(0.3)
       - Layer 2: Linear(256 → 2)   → logits [real, fake]
    5. Softmax → probability distribution
    6. Argmax → binary label  {0: Real, 1: Fake}

OUTPUT:
    - label     : int   {0: Real, 1: Fake}
    - confidence: float  (max softmax probability)

TRAINING:
    - Loss       : CrossEntropyLoss
    - Optimiser  : AdamW  (lr=2e-5, weight_decay=0.01)
    - Scheduler  : LinearWarmupScheduler (warmup=10% steps)
    - Epochs     : 5
    - Batch size : 32

EVALUATION METRICS:
    - Accuracy, F1-score (macro), Precision, Recall
    - Latency (ms/sample)  — no retrieval overhead

LIMITATIONS OF THIS BASELINE:
    - No visual signals (image/video ignored)
    - No external evidence or retrieval
    - No relational / propagation context
    - Black-box output — no explainability
========================================================================
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pandas as pd
import numpy as np
import time
import os


# ─────────────────────────────────────────────
# 1. DATASET LOADER
# ─────────────────────────────────────────────

class MisinformationTextDataset(Dataset):
    """
    Loads text-only samples from FakeNewsNet or Twitter datasets.

    Expected CSV columns:
        - 'text'  : the claim / headline / tweet body
        - 'label' : 0 = real, 1 = fake

    FakeNewsNet loading example:
        df = pd.read_csv("fakenewsnet_politifact.csv")

    Twitter loading example:
        df = pd.read_csv("twitter_misinfo.csv")
        df = df.rename(columns={"tweet_text": "text", "misinformation": "label"})
    """

    def __init__(self, dataframe: pd.DataFrame, tokenizer: BertTokenizer, max_length: int = 128):
        self.texts  = dataframe["text"].values.tolist()
        self.labels = dataframe["label"].values.tolist()
        self.tokenizer  = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
    "input_ids":      encoding["input_ids"].flatten(),
    "attention_mask": encoding["attention_mask"].flatten(),
    "label":          torch.tensor(self.labels[idx], dtype=torch.long),
}


# ─────────────────────────────────────────────
# 2. MODEL ARCHITECTURE
# ─────────────────────────────────────────────

class BaselineMisinformationDetector(nn.Module):
    """
    Single-agent BERT-based binary classifier.

    Architecture:
        BERT encoder  →  [CLS] pooling  →  MLP head  →  logits

    Input  : tokenised claim text
    Output : logits over {real, fake}
    """

    def __init__(self, bert_model_name: str = "bert-base-uncased",
                 hidden_dim: int = 256, num_classes: int = 2,
                 dropout: float = 0.3, freeze_bert_layers: int = 8):
        super().__init__()

        # ── Encoder ──────────────────────────────────────────────────
        self.bert = BertModel.from_pretrained(bert_model_name)

        # Freeze bottom N transformer layers to reduce compute
        for i, layer in enumerate(self.bert.encoder.layer):
            if i < freeze_bert_layers:
                for param in layer.parameters():
                    param.requires_grad = False

        # ── Classifier Head ──────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(768, hidden_dim),   # 768 = BERT hidden size
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        """
        Args:
            input_ids      : (batch, seq_len)
            attention_mask : (batch, seq_len)

        Returns:
            logits         : (batch, num_classes)
        """
        outputs    = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embed  = outputs.last_hidden_state[:, 0, :]  # (batch, 768)  [CLS] token
        logits     = self.classifier(cls_embed)           # (batch, 2)
        return logits


# ─────────────────────────────────────────────
# 3. TRAINING LOOP
# ─────────────────────────────────────────────

class BaselineTrainer:
    """Encapsulates training, evaluation, and latency benchmarking."""

    def __init__(self, model: BaselineMisinformationDetector,
                 train_loader: DataLoader, val_loader: DataLoader,
                 device: str = "cuda", lr: float = 2e-5,
                 weight_decay: float = 0.01, epochs: int = 5):

        self.model        = model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.device       = device
        self.epochs       = epochs

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, weight_decay=weight_decay
        )

        total_steps = len(train_loader) * epochs
        warmup_steps = int(0.1 * total_steps)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )

    def train_epoch(self):
        self.model.train()
        total_loss, all_preds, all_labels = 0.0, [], []

        for batch in self.train_loader:
            input_ids      = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels         = batch["label"].to(self.device)

            self.optimizer.zero_grad()
            logits = self.model(input_ids, attention_mask)
            loss   = self.criterion(logits, labels)
            loss.backward()

            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

        return {
            "loss":      total_loss / len(self.train_loader),
            "accuracy":  accuracy_score(all_labels, all_preds),
            "f1":        f1_score(all_labels, all_preds, average="macro"),
        }

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        all_preds, all_labels, latencies = [], [], []

        for batch in self.val_loader:
            input_ids      = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels         = batch["label"]

            t0     = time.perf_counter()
            logits = self.model(input_ids, attention_mask)
            latencies.append((time.perf_counter() - t0) * 1000 / len(labels))  # ms/sample

            preds = torch.argmax(logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

        return {
            "accuracy":        accuracy_score(all_labels, all_preds),
            "f1_macro":        f1_score(all_labels, all_preds, average="macro"),
            "precision_macro": precision_score(all_labels, all_preds, average="macro"),
            "recall_macro":    recall_score(all_labels, all_preds, average="macro"),
            "latency_ms":      float(np.mean(latencies)),
        }

    def fit(self):
        print("=" * 60)
        print("  MODEL 1 — BASELINE TRAINING")
        print("=" * 60)
        for epoch in range(1, self.epochs + 1):
            train_metrics = self.train_epoch()
            val_metrics   = self.evaluate()
            print(
                f"Epoch {epoch}/{self.epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Train Acc: {train_metrics['accuracy']:.4f} | "
                f"Val F1: {val_metrics['f1_macro']:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.4f} | "
                f"Latency: {val_metrics['latency_ms']:.2f} ms/sample"
            )
        return val_metrics


# ─────────────────────────────────────────────
# 4. INFERENCE  (single claim)
# ─────────────────────────────────────────────

class BaselineInference:
    """
    Wraps the trained model for single-claim real-time inference.

    Usage:
        engine = BaselineInference(model, tokenizer, device)
        result = engine.predict("Drinking bleach cures COVID-19")
        # → {"label": "Fake", "confidence": 0.97, "latency_ms": 8.3}
    """

    LABEL_MAP = {0: "Real", 1: "Fake"}

    def __init__(self, model: BaselineMisinformationDetector,
                 tokenizer: BertTokenizer, device: str = "cuda",
                 max_length: int = 128):
        self.model      = model.eval().to(device)
        self.tokenizer  = tokenizer
        self.device     = device
        self.max_length = max_length

    @torch.no_grad()
    def predict(self, claim: str) -> dict:
        encoding = self.tokenizer(
            claim,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids      = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        t0     = time.perf_counter()
        logits = self.model(input_ids, attention_mask)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        probs      = torch.softmax(logits, dim=1).squeeze(0)
        pred_class = torch.argmax(probs).item()
        confidence = probs[pred_class].item()

        return {
            "label":       self.LABEL_MAP[pred_class],
            "confidence":  round(confidence, 4),
            "latency_ms":  round(elapsed_ms, 2),
            "explanation": "N/A — Baseline model produces no explanation.",
        }


# ─────────────────────────────────────────────
# 5.  ENTRY POINT
# ─────────────────────────────────────────────

def load_combined_dataset(fakenewsnet_csv: str, twitter_csv: str) -> pd.DataFrame:
    """
    Merge FakeNewsNet + Twitter datasets into a unified dataframe.

    FakeNewsNet schema  : columns 'title'  (text) + 'label' (real=0, fake=1)
    Twitter schema      : columns 'text'           + 'label'
    """
    frames = []

    if os.path.exists(fakenewsnet_csv):
        df_fn = pd.read_csv(fakenewsnet_csv)
        df_fn = df_fn.rename(columns={"title": "text"})[["text", "label"]]
        frames.append(df_fn)

    if os.path.exists(twitter_csv):
        df_tw = pd.read_csv(twitter_csv)[["text", "label"]]
        frames.append(df_tw)

    if not frames:
        # ── Synthetic stub for smoke-testing ─────────────────────────
        print("[WARNING] No dataset CSVs found — using synthetic stub data.")
        stub = pd.DataFrame({
            "text":  [
                "Government confirms new vaccine is 99% effective.",
                "Scientists say drinking vinegar cures cancer instantly.",
                "UN releases annual climate change report.",
                "Aliens have landed in New York City says anonymous source.",
            ],
            "label": [0, 1, 0, 1],
        })
        return stub

    combined = pd.concat(frames, ignore_index=True).dropna(subset=["text", "label"])
    combined["label"] = combined["label"].astype(int)
    return combined.sample(frac=1, random_state=42).reset_index(drop=True)


def main():
    DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
    BERT_MODEL      = "bert-base-uncased"
    MAX_LEN         = 128
    BATCH_SIZE      = 32
    EPOCHS          = 5
    FAKENEWSNET_CSV = "data/fakenewsnet.csv"
    TWITTER_CSV     = "data/twitter_misinfo.csv"

    print(f"Device: {DEVICE}")

    # ── Data ─────────────────────────────────────────────────────────
    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
    df        = load_combined_dataset(FAKENEWSNET_CSV, TWITTER_CSV)

    split     = int(0.8 * len(df))
    train_df, val_df = df.iloc[:split], df.iloc[split:]

    train_ds  = MisinformationTextDataset(train_df, tokenizer, MAX_LEN)
    val_ds    = MisinformationTextDataset(val_df,   tokenizer, MAX_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # ── Model ────────────────────────────────────────────────────────
    model   = BaselineMisinformationDetector(bert_model_name=BERT_MODEL)
    trainer = BaselineTrainer(model, train_loader, val_loader,
                              device=DEVICE, epochs=EPOCHS)
    final_metrics = trainer.fit()

    print("\n── Final Validation Metrics ──────────────────────────────")
    for k, v in final_metrics.items():
        print(f"  {k:20s}: {v:.4f}")

    # ── Inference demo ───────────────────────────────────────────────
    engine = BaselineInference(model, tokenizer, DEVICE)
    test_claims = [
        "Scientists confirm 5G towers spread coronavirus.",
        "WHO publishes guidelines for pandemic preparedness.",
    ]
    print("\n── Inference Demo ────────────────────────────────────────")
    for claim in test_claims:
        result = engine.predict(claim)
        print(f"  Claim      : {claim}")
        print(f"  Label      : {result['label']}  (conf={result['confidence']:.2%})")
        print(f"  Latency    : {result['latency_ms']} ms")
        print(f"  Explanation: {result['explanation']}")
        print()

    # ── Save checkpoint ──────────────────────────────────────────────
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/model1_baseline.pt")
    print("Checkpoint saved → checkpoints/model1_baseline.pt")


if __name__ == "__main__":
    main()