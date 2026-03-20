"""
========================================================================
MODEL 3: GRAPH-RAG MODEL
========================================================================
Architecture  : Single-Agent with Retrieval + Graph-Based Context
Modality      : Multimodal (Text + Image)
Retrieval     : Dense Passage Retrieval (DPR) over evidence corpus
Graph Context : Claim propagation graph (PyG — Graph Attention Network)
Datasets      : FakeNewsNet, Twitter/Social Graph
Framework     : PyTorch + HuggingFace Transformers + PyG + FAISS
========================================================================

PSEUDOCODE / DESIGN SPEC
--------------------------
INPUT:
    - claim_text   (str)            : news headline / tweet body
    - claim_image  (PIL.Image|None) : optional accompanying image
    - claim_graph  (torch_geometric.data.Data):
          node_features : (N, 768)   — BERT embeddings of each node
                                        (claim + replies + retweets)
          edge_index    : (2, E)     — directed propagation edges
          edge_attr     : (E, 1)     — edge weights (retweet count /
                                        reply depth)

PIPELINE:

  STAGE 1 — MULTIMODAL ENCODING   [same as Model 2]
    1a. Text stream  : BERT  →  [CLS] embed (768)
    1b. Image stream : ViT   →  [CLS] embed (768)  | zeros if absent
    1c. Fusion       : concat → Linear(1536→768) → fused_embed (768)

  STAGE 2 — EVIDENCE RETRIEVAL    [same as Model 2]
    2a. Query FAISS with fused_embed  →  top-K passages (K=5)
    2b. Encode passages with BERT     →  evidence_embeds (K, 768)
    2c. Cross-attention fusion        →  rag_embed (768)

  STAGE 3 — GRAPH ENCODING  [NEW in Model 3]
    3a. Node init  : each node already has a 768-dim BERT embedding
    3b. GAT layer 1: GraphAttentionConv(768 → 256, heads=4)
                     multi-head attention over propagation edges
                     → node_feats_1 (N, 256)
    3c. GAT layer 2: GraphAttentionConv(256 → 128, heads=2)
                     → node_feats_2 (N, 128)  [root=claim node]
    3d. Global mean pool over all nodes → graph_embed (128)
    3e. Linear projection: graph_embed (128) → (768)

  STAGE 4 — GRAPH-RAG FUSION  [NEW in Model 3]
    4a. Concatenate [rag_embed | graph_proj]  →  (1536)
    4b. Linear(1536 → 768) + ReLU  →  joint_embed (768)

  STAGE 5 — CLASSIFICATION
    5a. joint_embed → MLP head → logits [real, fake]
    5b. Softmax → probabilities
    5c. Argmax  → binary label {0: Real, 1: Fake}

OUTPUT:
    - label            : int    {0: Real, 1: Fake}
    - confidence       : float
    - evidence_texts   : List[str]   top-K retrieved passages
    - graph_node_scores: List[float] per-node GAT attention weights
                                     (propagation explainability)
    - latency_ms       : float

GRAPH CONSTRUCTION (FakeNewsNet / Twitter):
    - Root node   = claim / original tweet
    - Child nodes = replies, retweets, quote-tweets
    - Edge        = directed from parent → child
    - Edge weight = retweet_count / (reply_depth + 1)
    - Node features = BERT([CLS]) of node text

TRAINING:
    - Loss       : CrossEntropyLoss
    - Optimiser  : AdamW  (lr=2e-5, weight_decay=0.01)
    - Scheduler  : LinearWarmupScheduler (warmup=10% steps)
    - Epochs     : 5
    - Batch size : 8  (graph batching overhead)

KEY IMPROVEMENTS OVER MODEL 2:
    ✓ Claim propagation modelled as a directed graph
    ✓ Graph Attention Network captures structural spread patterns
    ✓ Per-node attention weights expose HOW misinformation propagates
    ✓ Graph + RAG jointly enriches the final representation

REMAINING LIMITATIONS:
    ✗ Still a single agent — no specialised roles
    ✗ No inter-agent deliberation or reasoning chain
========================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from torch_geometric.nn import GATConv, global_mean_pool
from transformers import (
    BertTokenizer, BertModel,
    ViTFeatureExtractor, ViTModel,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from PIL import Image
from typing import Optional
import pandas as pd
import numpy as np
import faiss
import time
import os


# ─────────────────────────────────────────────
# 1. GRAPH CONSTRUCTION UTILITIES
# ─────────────────────────────────────────────

class ClaimGraphBuilder:
    """
    Converts a FakeNewsNet or Twitter propagation record into a
    PyTorch Geometric Data object.

    FakeNewsNet graph schema (expected dict):
        {
          "nodes": [
              {"id": 0, "text": "original claim …", "type": "root"},
              {"id": 1, "text": "reply text …",      "type": "reply"},
              ...
          ],
          "edges": [
              {"src": 0, "dst": 1, "weight": 1.0},
              ...
          ]
        }

    Twitter graph schema (expected dict):
        {
          "nodes": [
              {"id": 0, "text": "tweet …",   "type": "root"},
              {"id": 1, "text": "retweet …", "type": "retweet"},
              ...
          ],
          "edges": [
              {"src": 0, "dst": 1, "weight": 3.0},
              ...
          ]
        }
    """

    def __init__(self, bert_model: BertModel,
                 tokenizer: BertTokenizer,
                 device: str = "cuda",
                 max_nodes: int = 50,
                 max_text_len: int = 128):
        self.bert        = bert_model
        self.tokenizer   = tokenizer
        self.device      = device
        self.max_nodes   = max_nodes
        self.max_text_len = max_text_len

    @torch.no_grad()
    def _encode_node_texts(self, texts: list[str]) -> torch.Tensor:
        """Encode node texts → (N, 768) BERT [CLS] embeddings."""
        enc = self.tokenizer(
            texts, max_length=self.max_text_len,
            padding=True, truncation=True, return_tensors="pt"
        )
        out = self.bert(
            input_ids=enc["input_ids"].to(self.device),
            attention_mask=enc["attention_mask"].to(self.device),
        )
        return out.last_hidden_state[:, 0, :].cpu()   # (N, 768)

    def build(self, graph_dict: dict) -> Data:
        """
        Args:
            graph_dict : see schema above

        Returns:
            torch_geometric.data.Data with:
                x          : (N, 768)  node features
                edge_index : (2, E)    directed edges
                edge_attr  : (E, 1)    edge weights
        """
        nodes = graph_dict.get("nodes", [])[:self.max_nodes]
        edges = graph_dict.get("edges", [])

        if not nodes:
            # Fallback: single-node graph (root only)
            nodes = [{"id": 0, "text": "unknown claim", "type": "root"}]

        texts = [n["text"] for n in nodes]
        x     = self._encode_node_texts(texts)          # (N, 768)

        if edges:
            valid_ids  = {n["id"] for n in nodes}
            edges      = [e for e in edges
                          if e["src"] in valid_ids and e["dst"] in valid_ids]
            # Re-index node ids to [0, N-1]
            id_to_idx  = {n["id"]: i for i, n in enumerate(nodes)}
            src_list   = [id_to_idx[e["src"]] for e in edges]
            dst_list   = [id_to_idx[e["dst"]] for e in edges]
            weights    = [e.get("weight", 1.0) for e in edges]

            edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
            edge_attr  = torch.tensor(weights, dtype=torch.float).unsqueeze(1)
        else:
            # Self-loop on root node if no edges provided
            edge_index = torch.zeros(2, 1, dtype=torch.long)
            edge_attr  = torch.ones(1, 1, dtype=torch.float)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    @staticmethod
    def stub_graph(claim_text: str) -> dict:
        """
        Generates a minimal synthetic graph for smoke-testing
        when no real propagation data is available.
        """
        return {
            "nodes": [
                {"id": 0, "text": claim_text,                     "type": "root"},
                {"id": 1, "text": f"Reply: I agree — {claim_text[:40]}", "type": "reply"},
                {"id": 2, "text": f"RT @user: {claim_text[:40]}",        "type": "retweet"},
                {"id": 3, "text": "This is false information.",           "type": "reply"},
            ],
            "edges": [
                {"src": 0, "dst": 1, "weight": 1.0},
                {"src": 0, "dst": 2, "weight": 2.5},
                {"src": 0, "dst": 3, "weight": 1.0},
                {"src": 1, "dst": 2, "weight": 0.5},
            ],
        }


# ─────────────────────────────────────────────
# 2. DATASET  (multimodal + graph)
# ─────────────────────────────────────────────

class GraphRAGDataset(Dataset):
    """
    Loads text + image + propagation graph for each sample.

    Expected DataFrame columns:
        - 'text'       : claim / headline / tweet
        - 'label'      : 0 = real, 1 = fake
        - 'image_path' : path to image (may be NaN)
        - 'graph'      : dict following ClaimGraphBuilder schema
                         (may be NaN → stub graph used)
    """

    def __init__(self, dataframe: pd.DataFrame,
                 bert_tokenizer: BertTokenizer,
                 vit_extractor:  ViTFeatureExtractor,
                 graph_builder:  ClaimGraphBuilder,
                 max_length: int = 128):
        self.df            = dataframe.reset_index(drop=True)
        self.tokenizer     = bert_tokenizer
        self.vit_extractor = vit_extractor
        self.graph_builder = graph_builder
        self.max_length    = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row   = self.df.iloc[idx]
        text  = str(row["text"])
        label = int(row["label"])

        # ── Text ─────────────────────────────────────────────────────
        enc = self.tokenizer(
            text, max_length=self.max_length,
            padding="max_length", truncation=True, return_tensors="pt"
        )

        # ── Image ─────────────────────────────────────────────────────
        image_path = row.get("image_path", None)
        has_image  = False
        if pd.notna(image_path) and os.path.exists(str(image_path)):
            try:
                img          = Image.open(image_path).convert("RGB")
                vit_input    = self.vit_extractor(images=img, return_tensors="pt")
                pixel_values = vit_input["pixel_values"].squeeze(0)
                has_image    = True
            except Exception:
                pixel_values = torch.zeros(3, 224, 224)
        else:
            pixel_values = torch.zeros(3, 224, 224)

        # ── Graph ─────────────────────────────────────────────────────
        # ✅ NO GRAPH LEAKAGE (single-node graph)
        graph_dict = row.get("graph", None)

        if isinstance(graph_dict, dict):
            graph_data = self.graph_builder.build(graph_dict)
        else:
            graph_data = ClaimGraphBuilder.stub_graph(text)
            graph_data = self.graph_builder.build(graph_data)

        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "pixel_values":   pixel_values,
            "has_image":      torch.tensor(has_image, dtype=torch.bool),
            "graph":          graph_data,
            "label":          torch.tensor(label, dtype=torch.long),
        }


def graph_rag_collate_fn(batch: list[dict]) -> dict:
    """
    Custom collate: uses PyG Batch for graphs, standard stack for tensors.
    """
    graphs = Batch.from_data_list([item.pop("graph") for item in batch])
    out    = {k: torch.stack([b[k] for b in batch]) for k in batch[0]}
    out["graph"] = graphs
    return out


# ─────────────────────────────────────────────
# 3. GRAPH ATTENTION ENCODER
# ─────────────────────────────────────────────

class PropagationGraphEncoder(nn.Module):
    """
    Two-layer Graph Attention Network (GAT) over the claim
    propagation graph.

    Input  : node features (N, 768)
    Output : graph-level embedding (768) via global mean pool
             + per-node attention weights for explainability
    """

    def __init__(self, in_dim: int = 768, hidden_dim: int = 256,
                 out_dim: int = 128, heads1: int = 4, heads2: int = 2,
                 dropout: float = 0.5):
        super().__init__()

        self.gat1   = GATConv(in_dim,     hidden_dim,
                              heads=heads1, dropout=dropout,
                              concat=True)       # out: (N, hidden_dim * heads1)
        self.gat2   = GATConv(hidden_dim * heads1, out_dim,
                              heads=heads2, dropout=dropout,
                              concat=False)      # out: (N, out_dim)

        self.proj   = nn.Linear(out_dim, 768)
        self.norm   = nn.LayerNorm(768)
        self.drop   = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor,
                batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x          : (total_N, 768)  — all nodes across batched graphs
            edge_index : (2, total_E)
            batch      : (total_N,)      — node → graph mapping

        Returns:
            graph_embed        : (B, 768)
            node_attention_raw : (total_N, out_dim)  for interpretability
        """
        # Layer 1
        h1 = F.elu(self.gat1(x, edge_index))          # (total_N, 256*4)
        h1 = self.drop(h1)

        # Layer 2
        h2 = self.gat2(h1, edge_index)                 # (total_N, 128)
        h2 = F.elu(h2)

        # Graph-level pooling
        pooled      = global_mean_pool(h2, batch)       # (B, 128)
        graph_embed = self.norm(self.proj(pooled))       # (B, 768)

        return graph_embed, h2                           # h2 = per-node feats


# ─────────────────────────────────────────────
# 4. EVIDENCE CORPUS & FAISS (reused from Model 2)
# ─────────────────────────────────────────────

class EvidenceCorpus:
    def __init__(self, passages: list[str], bert_model: BertModel,
                 tokenizer: BertTokenizer, device: str = "cuda",
                 embed_dim: int = 768):
        self.passages  = passages
        self.bert      = bert_model
        self.tokenizer = tokenizer
        self.device    = device
        self.embed_dim = embed_dim
        self.index     = None

    @torch.no_grad()
    def _encode_passages(self, batch_size: int = 64) -> np.ndarray:
        self.bert.eval()
        all_embeds = []
        for i in range(0, len(self.passages), batch_size):
            batch = self.passages[i: i + batch_size]
            enc   = self.tokenizer(
                batch, max_length=256, padding=True,
                truncation=True, return_tensors="pt"
            )
            out = self.bert(
                input_ids=enc["input_ids"].to(self.device),
                attention_mask=enc["attention_mask"].to(self.device),
            )
            cls = out.last_hidden_state[:, 0, :].cpu().numpy()
            all_embeds.append(cls)
        embeds = np.vstack(all_embeds).astype("float32")
        faiss.normalize_L2(embeds)
        return embeds

    def build_index(self):
        print("[EvidenceCorpus] Building FAISS index …")
        embeds     = self._encode_passages()
        self.index = faiss.IndexFlatIP(self.embed_dim)
        self.index.add(embeds)
        print(f"[EvidenceCorpus] Index ready — {self.index.ntotal} passages.")

    def retrieve(self, query_embed: np.ndarray,
                 top_k: int = 5) -> list[dict]:
        assert self.index is not None, "Call build_index() first."
        q = query_embed.astype("float32")
        faiss.normalize_L2(q)
        scores, indices = self.index.search(q, top_k)
        return [
            {"passage": self.passages[idx], "score": float(sc)}
            for sc, idx in zip(scores[0], indices[0])
            if idx < len(self.passages)
        ]


class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim: int = 768, num_heads: int = 8):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, claim_embed: torch.Tensor,
                evidence_embeds: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attn(
            query=claim_embed,
            key=evidence_embeds,
            value=evidence_embeds,
        )
        return self.norm(attended.squeeze(1) + claim_embed.squeeze(1))


@torch.no_grad()
def encode_evidence_passages(passages: list[list[str]],
                              bert_model: BertModel,
                              tokenizer: BertTokenizer,
                              device: str,
                              max_len: int = 256) -> torch.Tensor:
    B, K = len(passages), len(passages[0]) if passages else 1
    all_embeds = torch.zeros(B, K, 768, device=device)
    bert_model.eval()
    for b, plist in enumerate(passages):
        if not plist:
            continue
        enc = tokenizer(
            plist, max_length=max_len, padding=True,
            truncation=True, return_tensors="pt"
        )
        out = bert_model(
            input_ids=enc["input_ids"].to(device),
            attention_mask=enc["attention_mask"].to(device),
        )
        all_embeds[b, :len(plist)] = out.last_hidden_state[:, 0, :]
    return all_embeds


# ─────────────────────────────────────────────
# 5. FULL MODEL ARCHITECTURE
# ─────────────────────────────────────────────

class GraphRAGDetector(nn.Module):
    """
    Single-agent Graph-RAG multimodal misinformation detector.

    Stages:
        1. BERT text encoder         →  text_embed  (768)
        2. ViT  image encoder        →  image_embed (768)
        3. Multimodal fusion         →  fused_embed (768)
        4. FAISS retrieval           →  evidence_embeds (K, 768)
        5. Cross-attention RAG fusion→  rag_embed   (768)
        6. GAT propagation encoder   →  graph_embed (768)
        7. Graph-RAG joint fusion    →  joint_embed (768)
        8. MLP classifier            →  logits      (2)
    """

    def __init__(self,
                 bert_model_name: str  = "bert-base-uncased",
                 vit_model_name:  str  = "google/vit-base-patch16-224",
                 gat_hidden:      int  = 256,
                 gat_out:         int  = 128,
                 hidden_dim:      int  = 256,
                 num_classes:     int  = 2,
                 dropout:         float = 0.3):
        super().__init__()

        # ── Encoders ─────────────────────────────────────────────────
        self.bert_encoder = BertModel.from_pretrained(bert_model_name)
        self.vit_encoder  = ViTModel.from_pretrained(vit_model_name)

        for param in self.vit_encoder.parameters():
            param.requires_grad = False

        # ── Multimodal fusion ─────────────────────────────────────────
        self.mm_fusion = nn.Sequential(
            nn.Linear(768 + 768, 768),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ── RAG cross-attention ───────────────────────────────────────
        self.rag_attn = CrossAttentionFusion(embed_dim=768, num_heads=8)

        # ── Graph encoder ─────────────────────────────────────────────
        self.graph_encoder = PropagationGraphEncoder(
            in_dim=768, hidden_dim=gat_hidden,
            out_dim=gat_out, dropout=dropout
        )

        # ── Graph-RAG joint fusion ────────────────────────────────────
        self.joint_fusion = nn.Sequential(
            nn.Linear(768 + 768, 768),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # ── Classifier head ───────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(768, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    # ── Modality encoders (reused in inference) ───────────────────────
    def encode_text(self, input_ids, attention_mask) -> torch.Tensor:
        out = self.bert_encoder(input_ids=input_ids,
                                attention_mask=attention_mask)
        return out.last_hidden_state[:, 0, :]                # (B, 768)

    def encode_image(self, pixel_values, has_image) -> torch.Tensor:
        B     = pixel_values.size(0)
        embed = torch.zeros(B, 768, device=pixel_values.device)
        if has_image.any():
            idx        = has_image.nonzero(as_tuple=True)[0]
            out        = self.vit_encoder(pixel_values=pixel_values[idx])
            embed[idx] = out.last_hidden_state[:, 0, :]
        return embed                                         # (B, 768)

    def forward(self,
                input_ids:       torch.Tensor,
                attention_mask:  torch.Tensor,
                pixel_values:    torch.Tensor,
                has_image:       torch.Tensor,
                evidence_embeds: torch.Tensor,    # (B, K, 768)
                graph:           Batch,           # PyG batched graph
                ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            logits       : (B, 2)
            node_feats   : (total_N, 128)   for attention visualisation
        """
        # Stage 1–3 : multimodal fusion
        text_embed  = self.encode_text(input_ids, attention_mask)    # (B,768)
        image_embed = self.encode_image(pixel_values, has_image)     # (B,768)
        fused       = self.mm_fusion(
            torch.cat([text_embed, image_embed], dim=-1)             # (B,1536)
        )                                                             # (B,768)

        # Stage 4–5 : RAG cross-attention
        rag_embed   = self.rag_attn(
            fused.unsqueeze(1),       # (B, 1, 768)
            evidence_embeds,          # (B, K, 768)
        )                             # (B, 768)

        # Stage 6 : graph encoding
        graph_embed, node_feats = self.graph_encoder(
            graph.x.to(input_ids.device),
            graph.edge_index.to(input_ids.device),
            graph.batch.to(input_ids.device),
        )                             # (B, 768),  (total_N, 128)

        # Stage 7 : joint Graph-RAG fusion
        joint = self.joint_fusion(
            torch.cat([rag_embed, graph_embed], dim=-1)              # (B,1536)
        )                                                             # (B,768)

        # Stage 8 : classify
        logits = self.classifier(joint)                              # (B, 2)
        return logits, node_feats


# ─────────────────────────────────────────────
# 6. TRAINING LOOP
# ─────────────────────────────────────────────

class GraphRAGTrainer:

    def __init__(self, model: GraphRAGDetector,
                 corpus: EvidenceCorpus,
                 train_loader: DataLoader,
                 val_loader:   DataLoader,
                 device:  str   = "cuda",
                 lr:      float = 2e-5,
                 weight_decay: float = 0.01,
                 epochs:  int   = 5,
                 top_k:   int   = 5):

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
        total_steps    = len(train_loader) * epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(0.1 * total_steps),
            num_training_steps=total_steps,
        )

    def _retrieve_evidence(self,
                           text_embeds: torch.Tensor) -> torch.Tensor:
        np_embeds    = text_embeds.detach().cpu().numpy()
        all_passages = []
        for i in range(np_embeds.shape[0]):
            results = self.corpus.retrieve(np_embeds[i:i+1], top_k=self.top_k)
            all_passages.append([r["passage"] for r in results])
        return encode_evidence_passages(
            all_passages, self.model.bert_encoder,
            self.corpus.tokenizer, self.device
        )                                           # (B, K, 768)

    def _run_batch(self, batch):
        input_ids      = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        pixel_values   = batch["pixel_values"].to(self.device)
        has_image      = batch["has_image"].to(self.device)
        labels         = batch["label"].to(self.device)
        graph = batch["graph"]
        graph = graph.to(self.device)

        with torch.no_grad():
            text_embeds = self.model.encode_text(input_ids, attention_mask)
        ev_embeds = self._retrieve_evidence(text_embeds)

        logits, _ = self.model(
            input_ids, attention_mask, pixel_values,
            has_image, ev_embeds, graph
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
            logits, _      = self._run_batch(batch)

            # Note: _run_batch calls torch.no_grad internally for retrieval,
            # but we need to override model to no_grad here as well.
            lat_ms = (time.perf_counter() - t0) * 1000 / logits.size(0)
            latencies.append(lat_ms)

            all_preds.extend(torch.argmax(logits, 1).cpu().numpy())
            all_labels.extend(batch["label"].cpu().numpy())

        return {
            "accuracy":        accuracy_score(all_labels, all_preds),
            "f1_macro":        f1_score(all_labels, all_preds, average="macro"),
            "precision_macro": precision_score(all_labels, all_preds, average="macro"),
            "recall_macro":    recall_score(all_labels, all_preds, average="macro"),
            "latency_ms":      float(np.mean(latencies)),
        }

    def fit(self):
        print("=" * 60)
        print("  MODEL 3 — GRAPH-RAG TRAINING")
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
# 7. INFERENCE  (single claim — with evidence + graph)
# ─────────────────────────────────────────────

class GraphRAGInference:
    """
    Real-time inference returning:
        - Binary label + confidence
        - Retrieved evidence passages        (RAG explainability)
        - Per-node GAT attention scores      (graph explainability)

    Usage:
        engine = GraphRAGInference(model, corpus, graph_builder,
                                   bert_tokenizer, vit_extractor, device)
        result = engine.predict(
            claim       = "5G spreads coronavirus",
            image       = None,
            graph_dict  = {...}   # or None → stub graph used
        )
    """

    LABEL_MAP = {0: "Real", 1: "Fake"}

    def __init__(self, model: GraphRAGDetector,
                 corpus: EvidenceCorpus,
                 graph_builder: ClaimGraphBuilder,
                 bert_tokenizer: BertTokenizer,
                 vit_extractor:  ViTFeatureExtractor,
                 device: str = "cuda",
                 top_k:  int = 5,
                 max_len: int = 128):

        self.model          = model.eval().to(device)
        self.corpus         = corpus
        self.graph_builder  = graph_builder
        self.bert_tokenizer = bert_tokenizer
        self.vit_extractor  = vit_extractor
        self.device         = device
        self.top_k          = top_k
        self.max_len        = max_len

    @torch.no_grad()
    def predict(self, claim: str,
            image: Optional[Image.Image] = None,
            graph_dict: Optional[dict] = None) -> dict:

        t0 = time.perf_counter()

        # ── Text ──────────────────────────────────────────────────────
        enc = self.bert_tokenizer(
            claim, max_length=self.max_len,
            padding="max_length", truncation=True, return_tensors="pt"
        )
        input_ids      = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)

        # ── Image ──────────────────────────────────────────────────────
        if image is not None:
            vit_in       = self.vit_extractor(
                images=image.convert("RGB"), return_tensors="pt")
            pixel_values = vit_in["pixel_values"].to(self.device)
            has_image    = torch.tensor([True], device=self.device)
        else:
            pixel_values = torch.zeros(1, 3, 224, 224, device=self.device)
            has_image    = torch.tensor([False], device=self.device)

        # ── Graph ──────────────────────────────────────────────────────
        if graph_dict is None:
            graph_dict = ClaimGraphBuilder.stub_graph(claim)
        graph_data = self.graph_builder.build(graph_dict)
        graph_data = graph_data.to(self.device)

        # Add batch vector (single graph → all nodes belong to graph 0)
        graph_data.batch = torch.zeros(
            graph_data.num_nodes, dtype=torch.long, device=self.device
        )
        graph_batch = Batch.from_data_list([graph_data])

        # ── Retrieve evidence ──────────────────────────────────────────
        text_embed = self.model.encode_text(input_ids, attention_mask)
        results    = self.corpus.retrieve(
            text_embed.cpu().numpy(), top_k=self.top_k
        )
        passages   = [r["passage"] for r in results]
        ev_embeds  = encode_evidence_passages(
            [passages], self.model.bert_encoder,
            self.bert_tokenizer, self.device
        )                                              # (1, K, 768)

        # ── Forward ────────────────────────────────────────────────────
        logits, node_feats = self.model(
            input_ids, attention_mask, pixel_values,
            has_image, ev_embeds, graph_batch
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000

        probs      = torch.softmax(logits, dim=1).squeeze(0)
        pred_class = torch.argmax(probs).item()
        confidence = probs[pred_class].item()

        # Per-node importance = L2 norm of node feature vector
        node_scores = node_feats.norm(dim=-1).cpu().tolist()

        return {
            "label":             self.LABEL_MAP[pred_class],
            "confidence":        round(confidence, 4),
            "evidence":          passages,
            "graph_node_scores": node_scores,
            "latency_ms":        round(elapsed_ms, 2),
        }


# ─────────────────────────────────────────────
# 8. ENTRY POINT
# ─────────────────────────────────────────────

def load_dataset(fakenewsnet_csv: str, twitter_csv: str) -> pd.DataFrame:
    frames = []

    def clean_df(df, text_col, source_name):
        if text_col in df.columns:
            df = df.rename(columns={text_col: "text"})

        df = df.loc[:, ~df.columns.duplicated()]

        for col in ["image_path", "graph"]:
            if col not in df.columns:
                df[col] = None

        df["source"] = source_name   # ✅ IMPORTANT

        return df[["text", "label", "image_path", "graph", "source"]]

    if os.path.exists(fakenewsnet_csv):
        df1 = pd.read_csv(fakenewsnet_csv)
        frames.append(clean_df(df1, "title", "fakenewsnet"))

    if os.path.exists(twitter_csv):
        df2 = pd.read_csv(twitter_csv)
        frames.append(clean_df(df2, "text", "twitter"))

    if not frames:
        print("[WARNING] No dataset found — using stub.")
        return pd.DataFrame({
            "text": ["Fake news example", "Real news example"],
            "label": [1, 0],
            "image_path": [None, None],
            "graph": [None, None],
            "source": ["stub", "stub"],
        })

    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(int)

    return df.reset_index(drop=True)

def main():
    DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
    BERT_MODEL      = "bert-base-uncased"
    VIT_MODEL       = "google/vit-base-patch16-224"
    MAX_LEN         = 128
    BATCH_SIZE      = 8
    EPOCHS          = 3
    TOP_K           = 5
    FAKENEWSNET_CSV = "data/fakenewsnet.csv"
    TWITTER_CSV     = "data/twitter_misinfo.csv"

    print(f"Device: {DEVICE}")

    bert_tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
    vit_extractor  = ViTFeatureExtractor.from_pretrained(VIT_MODEL)
    bert_shared    = BertModel.from_pretrained(BERT_MODEL).to(DEVICE)

    # ── Graph builder ─────────────────────────────────────────────────
    graph_builder = ClaimGraphBuilder(
        bert_model=bert_shared,
        tokenizer=bert_tokenizer,
        device=DEVICE,
    )

    # ── Dataset ───────────────────────────────────────────────────────
    # ── Dataset ───────────────────────────────────────────────────────
    df = load_dataset(FAKENEWSNET_CSV, TWITTER_CSV)

    # ✅ TEXT NORMALIZATION (VERY IMPORTANT)
    df["text"] = (
        df["text"]
        .astype(str)
        .str.lower()
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
    )

    # ✅ REMOVE LOW-INFO TEXT
    df = df[df["text"].str.len() > 30]

    # ✅ REMOVE DUPLICATES
    df = df.drop_duplicates(subset=["text"])

    # ✅ SPLIT FIRST (NO LEAKAGE)
    from sklearn.model_selection import train_test_split

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["label"],
        random_state=42
    )

    # ✅ OPTIONAL SAMPLING (AFTER SPLIT)
    train_df = train_df.sample(min(1000, len(train_df)), random_state=42)
    val_df   = val_df.sample(min(300, len(val_df)), random_state=42)

    train_ds = GraphRAGDataset(train_df, bert_tokenizer,
                               vit_extractor, graph_builder, MAX_LEN)
    val_ds   = GraphRAGDataset(val_df,   bert_tokenizer,
                               vit_extractor, graph_builder, MAX_LEN)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE,
                              shuffle=True,  collate_fn=graph_rag_collate_fn)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE,
                              shuffle=False, collate_fn=graph_rag_collate_fn)

    # ── Evidence corpus ───────────────────────────────────────────────
    texts = train_df["text"]

    # fix if it's accidentally a DataFrame
    if isinstance(texts, pd.DataFrame):
        texts = texts.iloc[:, 0]

    passages = texts.dropna().astype(str).sample(
        min(2000, len(texts)), random_state=42
    ).tolist()
    corpus = EvidenceCorpus(passages, bert_shared, bert_tokenizer, DEVICE)
    corpus.build_index()

    # ── Model ──────────────────────────────────────────────────────────
    model   = GraphRAGDetector(BERT_MODEL, VIT_MODEL)
    trainer = GraphRAGTrainer(
        model, corpus, train_loader, val_loader,
        device=DEVICE, epochs=EPOCHS, top_k=TOP_K
    )
    final_metrics = trainer.fit()

    print("\n── Final Validation Metrics ──────────────────────────────")
    for k, v in final_metrics.items():
        print(f"  {k:20s}: {v:.4f}")

    # ── Inference demo ────────────────────────────────────────────────
    engine = GraphRAGInference(
        model, corpus, graph_builder,
        bert_tokenizer, vit_extractor, DEVICE, top_k=TOP_K
    )
    print("\n── Inference Demo ────────────────────────────────────────")
    claim = "Scientists confirm 5G towers spread coronavirus."
    result = engine.predict(claim, image=None, graph_dict=None)

    print(f"  Claim             : {claim}")
    print(f"  Label             : {result['label']}  "
          f"(conf={result['confidence']:.2%})")
    print(f"  Latency           : {result['latency_ms']} ms")
    print(f"  Graph node scores : {[round(s,3) for s in result['graph_node_scores']]}")
    print(f"  Evidence:")
    for i, ev in enumerate(result["evidence"], 1):
        print(f"    [{i}] {ev[:110]}…")

    # ── Save checkpoint ───────────────────────────────────────────────
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/model3_graphrag.pt")
    print("\nCheckpoint saved → checkpoints/model3_graphrag.pt")


if __name__ == "__main__":
    main()