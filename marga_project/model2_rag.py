"""
========================================================================
MODEL 2: RAG-BASED MODEL
========================================================================
Architecture  : Single-Agent with Retrieval-Augmented Generation
Modality      : Multimodal (Text + Image)
Retrieval     : Dense Passage Retrieval (DPR) over evidence corpus
Graph Context : None
Datasets      : FakeNewsNet, Twitter/Social Graph (text + images)
Framework     : PyTorch + HuggingFace Transformers + FAISS
========================================================================

PSEUDOCODE / DESIGN SPEC
--------------------------
INPUT:
    - claim_text  (str)           : news headline / tweet body
    - claim_image (PIL.Image|None): optional accompanying image

PIPELINE:

  STAGE 1 — MULTIMODAL ENCODING
    1a. Text stream:
        - Tokenise claim_text  →  BERT encoder  →  [CLS] embed (768-dim)
    1b. Image stream (if image present):
        - Resize + normalise image
        - Pass through ViT (vision transformer) encoder  →  [CLS] embed (768-dim)
        - If no image: substitute zero vector (768-dim)
    1c. Fusion:
        - Concatenate [text_embed | image_embed]  →  (1536-dim)
        - Linear projection  →  fused_embed (768-dim)

  STAGE 2 — EVIDENCE RETRIEVAL  (RAG)
    2a. Encode fused_embed as query vector
    2b. FAISS nearest-neighbour search over pre-indexed evidence corpus
        - Corpus: FakeNewsNet article bodies + Twitter context threads
        - Top-K = 5 retrieved evidence passages
    2c. Encode each retrieved passage with BERT  →  evidence embeds

  STAGE 3 — EVIDENCE-CLAIM FUSION
    3a. Stack [fused_embed, evidence_1, ..., evidence_K]  →  (K+1, 768)
    3b. Cross-attention transformer layer:
        - Query  = fused_embed  (claim)
        - Key/Value = evidence embeds
        - Output = attended context (768-dim)
    3c. Residual add:  attended_context + fused_embed  →  enriched_embed

  STAGE 4 — CLASSIFICATION
    4a. enriched_embed  →  MLP head  →  logits [real, fake]
    4b. Softmax  →  probability distribution
    4c. Argmax  →  binary label {0: Real, 1: Fake}

OUTPUT:
    - label          : int    {0: Real, 1: Fake}
    - confidence     : float
    - evidence_texts : List[str]   top-K retrieved passages (explainability)
    - latency_ms     : float

TRAINING:
    - Loss       : CrossEntropyLoss
    - Optimiser  : AdamW  (lr=2e-5, weight_decay=0.01)
    - Scheduler  : LinearWarmupScheduler (warmup=10% steps)
    - Epochs     : 5
    - Batch size : 16  (reduced — multimodal overhead)

KEY IMPROVEMENTS OVER MODEL 1:
    ✓ Visual signals from claim images via ViT encoder
    ✓ External evidence grounding via FAISS retrieval
    ✓ Cross-attention fusion of claim vs. evidence
    ✓ Partial explainability via returned evidence passages

REMAINING LIMITATIONS:
    ✗ No relational / propagation graph context
    ✗ Single agent — no specialised roles (retriever vs. verifier)
    ✗ Evidence passages returned but not formally reasoned over
========================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import (
    BertTokenizer, BertModel,
    ViTFeatureExtractor, ViTModel,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from typing import Optional
from PIL import Image
import pandas as pd
import numpy as np
import faiss
import time
import os
import io


# ─────────────────────────────────────────────
# 1. EVIDENCE CORPUS & FAISS INDEX
# ─────────────────────────────────────────────

class EvidenceCorpus:
    """
    Builds and queries a FAISS flat-IP index over an evidence corpus.

    Corpus sources:
        - FakeNewsNet article bodies (real + fake)
        - Twitter context / reply threads

    Usage:
        corpus = EvidenceCorpus(passages, bert_model, tokenizer, device)
        corpus.build_index()
        results = corpus.retrieve(query_embedding, top_k=5)
    """

    def __init__(self, passages: list[str], bert_model: BertModel,
                 tokenizer: BertTokenizer, device: str = "cuda",
                 embed_dim: int = 768):
        self.passages   = passages
        self.bert       = bert_model
        self.tokenizer  = tokenizer
        self.device     = device
        self.embed_dim  = embed_dim
        self.index      = None

    @torch.no_grad()
    def _encode_passages(self, batch_size: int = 64) -> np.ndarray:
        """Encode all passages into L2-normalised BERT [CLS] vectors."""
        self.bert.eval()
        all_embeds = []

        for i in range(0, len(self.passages), batch_size):
            batch = self.passages[i: i + batch_size]
            enc   = self.tokenizer(
                batch, max_length=256, padding=True,
                truncation=True, return_tensors="pt"
            )
            out   = self.bert(
                input_ids=enc["input_ids"].to(self.device),
                attention_mask=enc["attention_mask"].to(self.device),
            )
            cls   = out.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeds.append(cls)

        embeds = np.vstack(all_embeds).astype("float32")
        faiss.normalize_L2(embeds)
        return embeds

    def build_index(self):
        print("[EvidenceCorpus] Encoding passages and building FAISS index …")
        embeds     = self._encode_passages()
        self.index = faiss.IndexFlatIP(self.embed_dim)   # inner-product (cosine after L2 norm)
        self.index.add(embeds)
        print(f"[EvidenceCorpus] Index built with {self.index.ntotal} passages.")

    def retrieve(self, query_embed: np.ndarray, top_k: int = 5) -> list[dict]:
        """
        Args:
            query_embed : (1, embed_dim) float32, already L2-normalised
            top_k       : number of passages to retrieve

        Returns:
            list of {"passage": str, "score": float}
        """
        assert self.index is not None, "Call build_index() first."
        query_embed = query_embed.astype("float32")
        faiss.normalize_L2(query_embed)
        scores, indices = self.index.search(query_embed, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.passages):
                results.append({"passage": self.passages[idx], "score": float(score)})
        return results


# ─────────────────────────────────────────────
# 2. DATASET  (multimodal)
# ─────────────────────────────────────────────

class MultimodalMisinformationDataset(Dataset):
    """
    Loads text + optional image pairs from FakeNewsNet / Twitter.

    Expected DataFrame columns:
        - 'text'      : claim / headline / tweet
        - 'label'     : 0 = real, 1 = fake
        - 'image_path': path to image file (may be NaN/empty → zero vector used)
    """

    def __init__(self, dataframe: pd.DataFrame,
                 bert_tokenizer: BertTokenizer,
                 vit_extractor: ViTFeatureExtractor,
                 max_length: int = 128):
        self.df            = dataframe.reset_index(drop=True)
        self.tokenizer     = bert_tokenizer
        self.vit_extractor = vit_extractor
        self.max_length    = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        text  = str(row["text"])
        label = int(row["label"])

        # ── Text encoding ────────────────────────────────────────────
        enc = self.tokenizer(
            text, max_length=self.max_length,
            padding="max_length", truncation=True, return_tensors="pt"
        )

        # ── Image encoding ───────────────────────────────────────────
        image_path = row.get("image_path", None)
        has_image  = False

        if pd.notna(image_path) and os.path.exists(str(image_path)):
            try:
                img       = Image.open(image_path).convert("RGB")
                vit_input = self.vit_extractor(images=img, return_tensors="pt")
                pixel_values = vit_input["pixel_values"].squeeze(0)   # (3, H, W)
                has_image = True
            except Exception:
                pixel_values = torch.zeros(3, 224, 224)
        else:
            pixel_values = torch.zeros(3, 224, 224)

        return {
            "input_ids": enc["input_ids"].flatten(),
            "attention_mask": enc["attention_mask"].flatten(),
            "pixel_values":   pixel_values,
            "has_image":      torch.tensor(has_image, dtype=torch.bool),
            "label":          torch.tensor(label, dtype=torch.long),
        }


# ─────────────────────────────────────────────
# 3. MODEL ARCHITECTURE
# ─────────────────────────────────────────────

class CrossAttentionFusion(nn.Module):
    """
    Single-head cross-attention:
        Query  = claim embedding
        Key/Value = stacked evidence embeddings

    Lets the claim 'attend' to the most relevant parts of retrieved evidence.
    """

    def __init__(self, embed_dim: int = 768, num_heads: int = 8):
        super().__init__()
        self.attn     = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm     = nn.LayerNorm(embed_dim)

    def forward(self, claim_embed: torch.Tensor,
                evidence_embeds: torch.Tensor) -> torch.Tensor:
        """
        Args:
            claim_embed     : (batch, 1, 768)
            evidence_embeds : (batch, K, 768)

        Returns:
            attended        : (batch, 768)
        """
        attended, _ = self.attn(
            query=claim_embed,
            key=evidence_embeds,
            value=evidence_embeds,
        )                                          # (batch, 1, 768)
        attended    = attended.squeeze(1)          # (batch, 768)
        claim_flat  = claim_embed.squeeze(1)       # (batch, 768)
        return self.norm(attended + claim_flat)    # residual + layer-norm


class RAGMisinformationDetector(nn.Module):
    """
    Single-agent RAG-based multimodal misinformation detector.

    Stages:
        1. Text  stream  : BERT  →  [CLS] (768)
        2. Image stream  : ViT   →  [CLS] (768)
        3. Fusion        : concat → Linear → (768)
        4. Retrieval     : FAISS evidence corpus  →  top-K passages
        5. Cross-attn    : claim attends to evidence  →  enriched (768)
        6. Classification: MLP → logits (2)
    """

    def __init__(self,
                 bert_model_name: str = "bert-base-uncased",
                 vit_model_name:  str = "google/vit-base-patch16-224",
                 hidden_dim:  int = 256,
                 num_classes: int = 2,
                 dropout:     float = 0.3,
                 num_heads:   int = 8):
        super().__init__()

        # ── Encoders ─────────────────────────────────────────────────
        self.bert_encoder = BertModel.from_pretrained(bert_model_name)
        self.vit_encoder  = ViTModel.from_pretrained(vit_model_name)

        # Freeze ViT (use as frozen feature extractor to save memory)
        for param in self.vit_encoder.parameters():
            param.requires_grad = False

        # ── Multimodal Fusion ─────────────────────────────────────────
        self.fusion_proj = nn.Sequential(
            nn.Linear(768 + 768, 768),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ── Evidence Cross-Attention ──────────────────────────────────
        self.evidence_attn = CrossAttentionFusion(embed_dim=768, num_heads=num_heads)

        # Evidence passage encoder (shares BERT weights)
        self.evidence_proj = nn.Linear(768, 768)

        # ── Classifier Head ───────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(768, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def encode_text(self, input_ids, attention_mask) -> torch.Tensor:
        out = self.bert_encoder(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:, 0, :]           # (B, 768)

    def encode_image(self, pixel_values, has_image) -> torch.Tensor:
        """Returns ViT [CLS] for real images; zero vector for missing images."""
        batch_size = pixel_values.size(0)
        embed = torch.zeros(batch_size, 768, device=pixel_values.device)

        if has_image.any():
            idx    = has_image.nonzero(as_tuple=True)[0]
            out    = self.vit_encoder(pixel_values=pixel_values[idx])
            embed[idx] = out.last_hidden_state[:, 0, :]  # (n_images, 768)
        return embed                                      # (B, 768)

    def forward(self,
                input_ids:       torch.Tensor,
                attention_mask:  torch.Tensor,
                pixel_values:    torch.Tensor,
                has_image:       torch.Tensor,
                evidence_embeds: torch.Tensor,            # (B, K, 768)  pre-encoded
                ) -> torch.Tensor:
        """
        Returns:
            logits : (B, num_classes)
        """
        # Stage 1–2: encode text and image
        text_embed  = self.encode_text(input_ids, attention_mask)   # (B, 768)
        image_embed = self.encode_image(pixel_values, has_image)    # (B, 768)

        # Stage 3: fuse modalities
        fused = self.fusion_proj(
            torch.cat([text_embed, image_embed], dim=-1)             # (B, 1536)
        )                                                             # (B, 768)

        # Stage 5: cross-attention over evidence
        enriched = self.evidence_attn(
            fused.unsqueeze(1),      # (B, 1, 768)  — query
            evidence_embeds,         # (B, K, 768)  — keys/values
        )                            # (B, 768)

        # Stage 6: classify
        logits = self.classifier(enriched)                           # (B, 2)
        return logits


# ─────────────────────────────────────────────
# 4. EVIDENCE EMBEDDING HELPER
# ─────────────────────────────────────────────

@torch.no_grad()
def encode_evidence_passages(passages: list[list[str]],
                              bert_model: BertModel,
                              tokenizer: BertTokenizer,
                              device: str,
                              max_len: int = 256) -> torch.Tensor:
    """
    Encode a batch of retrieved evidence passage-lists into tensors.

    Args:
        passages : list of K passages per sample  →  shape (B, K)
        returns  : (B, K, 768)
    """
    batch_size = len(passages)
    K          = len(passages[0]) if passages else 1
    all_embeds = torch.zeros(batch_size, K, 768, device=device)

    bert_model.eval()
    for b, plist in enumerate(passages):
        if not plist:
            continue
        enc = tokenizer(
            plist, max_length=max_len, padding=True,
            truncation=True, return_tensors="pt"
        )
        out    = bert_model(
            input_ids=enc["input_ids"].to(device),
            attention_mask=enc["attention_mask"].to(device),
        )
        embeds = out.last_hidden_state[:, 0, :]       # (K, 768)
        all_embeds[b, :len(plist)] = embeds

    return all_embeds                                  # (B, K, 768)


# ─────────────────────────────────────────────
# 5. TRAINING LOOP
# ─────────────────────────────────────────────

class RAGTrainer:
    """Training loop that integrates FAISS retrieval during forward pass."""

    def __init__(self, model: RAGMisinformationDetector,
                 corpus: EvidenceCorpus,
                 train_loader: DataLoader,
                 val_loader:   DataLoader,
                 device: str   = "cuda",
                 lr:    float  = 2e-5,
                 weight_decay: float = 0.01,
                 epochs: int   = 5,
                 top_k:  int   = 5):

        self.model        = model.to(device)
        self.corpus       = corpus
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.device       = device
        self.epochs       = epochs
        self.top_k        = top_k

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=lr, weight_decay=weight_decay
        )
        total_steps  = len(train_loader) * epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps
        )

    def _retrieve_evidence(self, text_embeds: torch.Tensor) -> torch.Tensor:
        """
        For each sample in the batch, retrieve top-K evidence passages
        and return their BERT embeddings: (B, K, 768).
        """
        np_embeds = text_embeds.detach().cpu().numpy()      # (B, 768)
        all_passages = []
        for i in range(np_embeds.shape[0]):
            results = self.corpus.retrieve(np_embeds[i:i+1], top_k=self.top_k)
            all_passages.append([r["passage"] for r in results])

        ev_embeds = encode_evidence_passages(
            all_passages, self.model.bert_encoder,
            self.corpus.tokenizer, self.device
        )
        return ev_embeds                                     # (B, K, 768)

    def _run_batch(self, batch):
        input_ids      = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        pixel_values   = batch["pixel_values"].to(self.device)
        has_image      = batch["has_image"].to(self.device)
        labels         = batch["label"].to(self.device)

        # Encode text for retrieval query
        with torch.no_grad():
            text_embeds = self.model.encode_text(input_ids, attention_mask)

        # Retrieve evidence
        ev_embeds = self._retrieve_evidence(text_embeds)    # (B, K, 768)

        # Forward pass
        logits = self.model(
            input_ids, attention_mask, pixel_values, has_image, ev_embeds
        )
        return logits, labels

    def train_epoch(self):
        self.model.train()
        total_loss, all_preds, all_labels = 0.0, [], []

        for batch in self.train_loader:
            logits, labels = self._run_batch(batch)
            loss = self.criterion(logits, labels)

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            all_preds.extend(torch.argmax(logits, 1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        return {
            "loss":     total_loss / len(self.train_loader),
            "accuracy": accuracy_score(all_labels, all_preds),
            "f1":       f1_score(all_labels, all_preds, average="macro"),
        }

    @torch.no_grad()
    def evaluate(self):
        self.model.eval()
        all_preds, all_labels, latencies = [], [], []

        for batch in self.val_loader:
            t0             = time.perf_counter()
            logits, labels = self._run_batch(batch)
            lat_ms         = (time.perf_counter() - t0) * 1000 / len(labels)
            latencies.append(lat_ms)

            all_preds.extend(torch.argmax(logits, 1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        return {
            "accuracy":        accuracy_score(all_labels, all_preds),
            "f1_macro":        f1_score(all_labels, all_preds, average="macro"),
            "precision_macro": precision_score(all_labels, all_preds, average="macro"),
            "recall_macro":    recall_score(all_labels, all_preds, average="macro"),
            "latency_ms":      float(np.mean(latencies)),
        }

    def fit(self):
        print("=" * 60)
        print("  MODEL 2 — RAG-BASED TRAINING")
        print("=" * 60)
        for epoch in range(1, self.epochs + 1):
            train_m = self.train_epoch()
            val_m   = self.evaluate()
            print(
                f"Epoch {epoch}/{self.epochs} | "
                f"Train Loss: {train_m['loss']:.4f} | "
                f"Train Acc: {train_m['accuracy']:.4f} | "
                f"Val F1: {val_m['f1_macro']:.4f} | "
                f"Val Acc: {val_m['accuracy']:.4f} | "
                f"Latency: {val_m['latency_ms']:.2f} ms/sample"
            )
        return val_m


# ─────────────────────────────────────────────
# 6. INFERENCE  (single claim — with evidence)
# ─────────────────────────────────────────────

class RAGInference:
    """
    Real-time inference with retrieved evidence passages as explainability.

    Usage:
        engine = RAGInference(model, corpus, bert_tokenizer,
                              vit_extractor, device)
        result = engine.predict("Bleach cures COVID", image=None)
        # → {
        #     "label": "Fake",
        #     "confidence": 0.95,
        #     "evidence": ["passage 1 …", "passage 2 …", …],
        #     "latency_ms": 42.1
        #   }
    """

    LABEL_MAP = {0: "Real", 1: "Fake"}

    def __init__(self, model: RAGMisinformationDetector,
                 corpus: EvidenceCorpus,
                 bert_tokenizer: BertTokenizer,
                 vit_extractor:  ViTFeatureExtractor,
                 device: str = "cuda",
                 top_k:  int = 5,
                 max_len: int = 128):

        self.model         = model.eval().to(device)
        self.corpus        = corpus
        self.bert_tokenizer = bert_tokenizer
        self.vit_extractor  = vit_extractor
        self.device        = device
        self.top_k         = top_k
        self.max_len       = max_len

    @torch.no_grad()
    def predict(self, claim: str,
               image: Optional[Image.Image] = None) -> dict:

        t0 = time.perf_counter()

        # ── Text ─────────────────────────────────────────────────────
        enc = self.bert_tokenizer(
            claim, max_length=self.max_len,
            padding="max_length", truncation=True, return_tensors="pt"
        )
        input_ids      = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        # ── Image ─────────────────────────────────────────────────────
        if image is not None:
            vit_in       = self.vit_extractor(images=image.convert("RGB"),
                                              return_tensors="pt")
            pixel_values = vit_in["pixel_values"].to(self.device)
            has_image    = torch.tensor([True], device=self.device)
        else:
            pixel_values = torch.zeros(1, 3, 224, 224, device=self.device)
            has_image    = torch.tensor([False], device=self.device)

        # ── Retrieve evidence ─────────────────────────────────────────
        text_embed = self.model.encode_text(input_ids, attention_mask)
        results    = self.corpus.retrieve(
            text_embed.cpu().numpy(), top_k=self.top_k
        )
        passages   = [r["passage"] for r in results]
        ev_embeds  = encode_evidence_passages(
            [passages], self.model.bert_encoder,
            self.bert_tokenizer, self.device
        )                                                  # (1, K, 768)

        # ── Forward ───────────────────────────────────────────────────
        logits     = self.model(
            input_ids, attention_mask, pixel_values, has_image, ev_embeds
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        probs      = torch.softmax(logits, dim=1).squeeze(0)
        pred_class = torch.argmax(probs).item()
        confidence = probs[pred_class].item()

        return {
            "label":      self.LABEL_MAP[pred_class],
            "confidence": round(confidence, 4),
            "evidence":   passages,
            "latency_ms": round(elapsed_ms, 2),
        }


# ─────────────────────────────────────────────
# 7. ENTRY POINT
# ─────────────────────────────────────────────

def load_dataset(fakenewsnet_csv: str, twitter_csv: str) -> pd.DataFrame:
    frames = []
    if os.path.exists(fakenewsnet_csv):
        df = pd.read_csv(fakenewsnet_csv)
        df = df.rename(columns={"title": "text"})
        if "image_path" not in df.columns:
            df["image_path"] = None
        frames.append(df[["text", "label", "image_path"]])

    if os.path.exists(twitter_csv):
        df = pd.read_csv(twitter_csv)
        if "image_path" not in df.columns:
            df["image_path"] = None
        frames.append(df[["text", "label", "image_path"]])

    if not frames:
        print("[WARNING] No dataset CSVs found — using synthetic stub.")
        return pd.DataFrame({
            "text": [
                "Government confirms new vaccine is 99% effective.",
                "Scientists say drinking vinegar cures cancer instantly.",
                "UN releases annual climate change report.",
                "Aliens have landed in New York City says anonymous source.",
            ],
            "label":      [0, 1, 0, 1],
            "image_path": [None, None, None, None],
        })

    df = pd.concat(frames, ignore_index=True).dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(int)
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


def build_evidence_corpus(df: pd.DataFrame, n_passages: int = 2000) -> list[str]:
    """Use the training texts themselves as a simple evidence corpus."""
    
    texts = df["text"].astype(str).values.tolist()

    clean_texts = []
    for t in texts:
        if isinstance(t, list):
            t = " ".join(map(str, t))   # convert list → string
        t = str(t).strip()              # ✅ ALWAYS apply
        if t != "":
            clean_texts.append(t)

    # sample safely
    sample_size = min(n_passages, len(clean_texts))
    np.random.seed(42)
    sampled = np.random.choice(clean_texts, sample_size, replace=False)

    return list(sampled)

def main():
    
    DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
    BERT_MODEL      = "bert-base-uncased"
    VIT_MODEL       = "google/vit-base-patch16-224"
    MAX_LEN         = 128
    BATCH_SIZE      = 8
    EPOCHS          = 5
    TOP_K           = 5
    FAKENEWSNET_CSV = "data/fakenewsnet.csv"
    TWITTER_CSV     = "data/twitter_misinfo.csv"

    print(f"Device: {DEVICE}")

    # ── Tokenizers / extractors ───────────────────────────────────────
    bert_tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
    vit_extractor  = ViTFeatureExtractor.from_pretrained(VIT_MODEL)

    # ── Dataset ───────────────────────────────────────────────────────
    df    = load_dataset(FAKENEWSNET_CSV, TWITTER_CSV)
    df = df.sample(min(5000, len(df)), random_state=42)  
    split = int(0.8 * len(df))
    train_df, val_df = df.iloc[:split], df.iloc[split:]

    train_ds = MultimodalMisinformationDataset(
        train_df, bert_tokenizer, vit_extractor, MAX_LEN)
    val_ds   = MultimodalMisinformationDataset(
        val_df, bert_tokenizer, vit_extractor, MAX_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    # ── Evidence corpus & FAISS index ────────────────────────────────
    bert_for_corpus = BertModel.from_pretrained(BERT_MODEL).to(DEVICE)
    passages = build_evidence_corpus(train_df, n_passages=1000)
    corpus          = EvidenceCorpus(
        passages, bert_for_corpus, bert_tokenizer, DEVICE
    )
    corpus.build_index()

    # ── Model ─────────────────────────────────────────────────────────
    model   = RAGMisinformationDetector(BERT_MODEL, VIT_MODEL)
    trainer = RAGTrainer(
        model, corpus, train_loader, val_loader,
        device=DEVICE, epochs=EPOCHS, top_k=TOP_K
    )
    final_metrics = trainer.fit()

    print("\n── Final Validation Metrics ──────────────────────────────")
    for k, v in final_metrics.items():
        print(f"  {k:20s}: {v:.4f}")

    # ── Inference demo ────────────────────────────────────────────────
    engine = RAGInference(
        model, corpus, bert_tokenizer, vit_extractor, DEVICE, top_k=TOP_K
    )
    test_claims = [
        ("Scientists confirm 5G towers spread coronavirus.", None),
        ("WHO publishes guidelines for pandemic preparedness.", None),
    ]
    print("\n── Inference Demo ────────────────────────────────────────")
    for claim, img in test_claims:
        result = engine.predict(claim, image=img)
        print(f"  Claim      : {claim}")
        print(f"  Label      : {result['label']}  (conf={result['confidence']:.2%})")
        print(f"  Latency    : {result['latency_ms']} ms")
        print(f"  Evidence   :")
        for i, ev in enumerate(result["evidence"], 1):
            print(f"    [{i}] {ev[:120]}…")
        print()

    # ── Save checkpoint ───────────────────────────────────────────────
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/model2_rag.pt")
    print("Checkpoint saved → checkpoints/model2_rag.pt")


if __name__ == "__main__":
    main()