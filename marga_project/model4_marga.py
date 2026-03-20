"""
========================================================================
MODEL 4: MARGA — Multi-Agent Retrieval-Augmented Graph Architecture
========================================================================
Architecture  : Multi-Agent System (Retriever + Verifier + Reasoner
                + Graph Analyst + Decision Aggregator)
Modality      : Multimodal (Text + Image)
Retrieval     : Dense Passage Retrieval (DPR) — FAISS
Graph Context : Claim propagation graph (GAT — PyTorch Geometric)
Datasets      : FakeNewsNet, Twitter/Social Graph
Framework     : PyTorch + HuggingFace Transformers + PyG + FAISS
========================================================================

PSEUDOCODE / DESIGN SPEC
--------------------------

INPUT:
    - claim_text   (str)            : news headline / tweet body
    - claim_image  (PIL.Image|None) : optional accompanying image
    - claim_graph  (PyG Data)       : propagation graph
                                      nodes = BERT-encoded texts
                                      edges = directed propagation links

═══════════════════════════════════════════════════════════════════════
AGENT DEFINITIONS
═══════════════════════════════════════════════════════════════════════

AGENT 1 — RETRIEVER AGENT
    Role     : Evidence acquisition specialist
    Input    : claim_text (raw)
    Process  :
        - Encode claim with DPR query encoder  →  query_vec (768)
        - FAISS top-K search over evidence corpus  →  K passages
        - Score each passage for relevance
    Output   : retrieved_passages (List[str]), relevance_scores (List[float])

AGENT 2 — VERIFIER AGENT
    Role     : Cross-modal claim-evidence consistency checker
    Input    : claim_text, claim_image, retrieved_passages
    Process  :
        - Encode text  via BERT   →  text_embed  (768)
        - Encode image via ViT    →  image_embed (768)
        - Encode each passage     →  ev_embeds   (K, 768)
        - Cross-modal attention   →  verified_embed (768)
        - Compute alignment score : cosine(text_embed, image_embed)
    Output   : verified_embed (768), cross_modal_score (float)

AGENT 3 — GRAPH ANALYST AGENT
    Role     : Propagation structure analyser
    Input    : claim_graph (PyG Data)
    Process  :
        - 2-layer GAT over propagation graph
        - Compute node-level attention weights
        - Global mean pool  →  structural_embed (768)
        - Derive propagation_risk_score from graph topology
    Output   : structural_embed (768), node_attention_weights (List[float])
               propagation_risk (float)

AGENT 4 — REASONER AGENT
    Role     : Deliberative reasoning and evidence synthesis
    Input    : verified_embed, structural_embed, retrieved_passages
    Process  :
        - Concatenate [verified_embed | structural_embed]  →  (1536)
        - Linear projection  →  joint_context (768)
        - Self-attention transformer block over joint context
        - Generate reasoning trace (attention-based salience)
    Output   : reasoned_embed (768), reasoning_trace (dict)

AGENT 5 — DECISION AGGREGATOR AGENT
    Role     : Final verdict synthesis with confidence calibration
    Input    : outputs from all 4 agents
    Process  :
        - Weighted ensemble:
            w_verifier  * verified_embed     (dynamic weight)
          + w_graph     * structural_embed   (dynamic weight)
          + w_reasoner  * reasoned_embed     (dynamic weight)
        - Weights computed via learned softmax gating network
        - MLP classifier  →  logits [real, fake]
        - Temperature scaling for calibrated confidence
    Output   : label, confidence, explanation chain

═══════════════════════════════════════════════════════════════════════
INTER-AGENT COMMUNICATION PROTOCOL
═══════════════════════════════════════════════════════════════════════
    - Agents communicate via a shared AgentContext message bus
    - Each agent reads from and writes to the context in sequence
    - Aggregator reads all agent outputs simultaneously
    - No feedback loops in base MARGA (feedforward only)

PIPELINE (sequential):
    claim + image + graph
         │
    [RETRIEVER] ──→ passages + scores
         │
    [VERIFIER]  ──→ verified_embed + cross_modal_score
         │
    [GRAPH ANALYST] ──→ structural_embed + node_weights + risk_score
         │
    [REASONER]  ──→ reasoned_embed + reasoning_trace
         │
    [AGGREGATOR] ──→ label + confidence + explanation_chain

OUTPUT:
    - label              : int    {0: Real, 1: Fake}
    - confidence         : float  (calibrated)
    - evidence_passages  : List[str]    top-K retrieved passages
    - reasoning_trace    : dict         salience map per token / node
    - node_attention     : List[float]  per-node GAT weights
    - propagation_risk   : float        graph topology risk score
    - cross_modal_score  : float        text-image alignment
    - agent_weights      : dict         learned per-agent contribution
    - latency_ms         : float

TRAINING:
    - Loss       : CrossEntropyLoss  +  λ * consistency_regulariser
                   (penalises high cross-modal misalignment on true samples)
    - Optimiser  : AdamW  (lr=2e-5, weight_decay=0.01)
    - Scheduler  : LinearWarmupScheduler (warmup=10% steps)
    - Epochs     : 5
    - Batch size : 8

KEY IMPROVEMENTS OVER MODEL 3:
    ✓ Specialised agents — each optimised for a distinct sub-task
    ✓ Learned dynamic weighting of agent contributions
    ✓ Temperature-scaled calibrated confidence output
    ✓ Full explanation chain: evidence + reasoning + graph + cross-modal
    ✓ Consistency regularisation aligns text-image representations
    ✓ Propagation risk score derived from graph topology metrics
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
    ViTImageProcessor, ViTModel,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from dataclasses import dataclass, field
from typing import Optional
from PIL import Image
import pandas as pd
import numpy as np
import faiss
import time
import os


# ─────────────────────────────────────────────
# 1. AGENT CONTEXT  (shared message bus)
# ─────────────────────────────────────────────

@dataclass
class AgentContext:
    """
    Shared message bus passed between agents.
    Each agent reads its required inputs and writes its outputs here.
    Acts as the single source of truth for the full inference pipeline.
    """

    # ── Raw inputs ────────────────────────────────────────────────────
    input_ids:      torch.Tensor = None
    attention_mask: torch.Tensor = None
    pixel_values:   torch.Tensor = None
    has_image:      torch.Tensor = None
    graph:          Batch        = None

    # ── Agent 1: Retriever outputs ────────────────────────────────────
    retrieved_passages:  list = field(default_factory=list)
    relevance_scores:    list = field(default_factory=list)
    evidence_embeds:     torch.Tensor = None     # (B, K, 768)

    # ── Agent 2: Verifier outputs ─────────────────────────────────────
    verified_embed:      torch.Tensor = None     # (B, 768)
    cross_modal_score:   torch.Tensor = None     # (B,)
    text_embed:          torch.Tensor = None     # (B, 768)  — shared downstream

    # ── Agent 3: Graph Analyst outputs ───────────────────────────────
    structural_embed:    torch.Tensor = None     # (B, 768)
    node_attention:      torch.Tensor = None     # (total_N, 128)
    propagation_risk:    torch.Tensor = None     # (B,)

    # ── Agent 4: Reasoner outputs ─────────────────────────────────────
    reasoned_embed:      torch.Tensor = None     # (B, 768)
    reasoning_trace:     dict         = field(default_factory=dict)

    # ── Agent 5: Aggregator outputs ───────────────────────────────────
    logits:              torch.Tensor = None     # (B, 2)
    agent_weights:       torch.Tensor = None     # (B, 3)  per-agent gate


# ─────────────────────────────────────────────
# 2. AGENT 1 — RETRIEVER
# ─────────────────────────────────────────────

class RetrieverAgent(nn.Module):
    """
    Encodes the claim and retrieves the top-K most relevant evidence
    passages from the FAISS index.

    Responsibilities:
        - DPR-style query encoding (BERT [CLS])
        - FAISS nearest-neighbour search
        - Evidence passage BERT encoding  →  evidence_embeds
    """

    def __init__(self, bert_model: BertModel,
                 tokenizer: BertTokenizer,
                 corpus_passages: list[str],
                 device: str = "cuda",
                 top_k: int = 5,
                 embed_dim: int = 768):
        super().__init__()
        self.bert      = bert_model
        self.tokenizer = tokenizer
        self.passages  = corpus_passages
        self.device    = device
        self.top_k     = top_k
        self.embed_dim = embed_dim
        self.index     = None

    @torch.no_grad()
    def build_index(self, batch_size: int = 64):
        print("[RetrieverAgent] Building FAISS evidence index …")
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
        self.index = faiss.IndexFlatIP(self.embed_dim)
        self.index.add(embeds)
        print(f"[RetrieverAgent] Index ready — {self.index.ntotal} passages.")

    @torch.no_grad()
    def _retrieve(self, query_np: np.ndarray) -> tuple[list[str], list[float]]:
        query_np = query_np.astype("float32")
        faiss.normalize_L2(query_np)
        scores, indices = self.index.search(query_np, self.top_k)
        passages, rel_scores = [], []
        for sc, idx in zip(scores[0], indices[0]):
            if idx < len(self.passages):
                passages.append(self.passages[idx])
                rel_scores.append(float(sc))
        return passages, rel_scores

    @torch.no_grad()
    def _encode_passages(self, passages: list[str],
                         max_len: int = 256) -> torch.Tensor:
        """Returns (K, 768) passage embeddings."""
        enc = self.tokenizer(
            passages, max_length=max_len,
            padding=True, truncation=True, return_tensors="pt"
        )
        out = self.bert(
            input_ids=enc["input_ids"].to(self.device),
            attention_mask=enc["attention_mask"].to(self.device),
        )
        return out.last_hidden_state[:, 0, :]     # (K, 768)

    def forward(self, ctx: AgentContext) -> AgentContext:
        """
        Reads  : ctx.text_embed  (B, 768)  — set by Verifier or pre-computed
        Writes : ctx.retrieved_passages, ctx.relevance_scores,
                 ctx.evidence_embeds (B, K, 768)
        """
        assert self.index is not None, "Call build_index() first."
        B           = ctx.text_embed.size(0)
        query_np    = ctx.text_embed.detach().cpu().numpy()   # (B, 768)

        all_passages, all_scores, all_ev_embeds = [], [], []

        for i in range(B):
            passages, scores = self._retrieve(query_np[i:i+1])
            ev_emb           = self._encode_passages(passages)   # (K, 768)
            all_passages.append(passages)
            all_scores.append(scores)
            all_ev_embeds.append(ev_emb)

        ctx.retrieved_passages = all_passages
        ctx.relevance_scores   = all_scores
        ctx.evidence_embeds    = torch.stack(all_ev_embeds)   # (B, K, 768)
        return ctx


# ─────────────────────────────────────────────
# 3. AGENT 2 — VERIFIER
# ─────────────────────────────────────────────

class VerifierAgent(nn.Module):
    """
    Cross-modal claim-evidence consistency verifier.

    Responsibilities:
        - Encode text (BERT) and image (ViT) modalities
        - Cross-attention fusion: claim attends to evidence
        - Compute text-image alignment score
        - Output verified multimodal embedding
    """

    def __init__(self, bert_model: BertModel,
                 vit_model: ViTModel,
                 dropout: float = 0.3,
                 num_heads: int = 8):
        super().__init__()
        self.bert       = bert_model
        self.vit        = vit_model

        # Freeze ViT
        for p in self.vit.parameters():
            p.requires_grad = False

        # Multimodal fusion
        self.mm_fusion = nn.Sequential(
            nn.Linear(768 + 768, 768),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Cross-attention: claim queries evidence
        self.cross_attn = nn.MultiheadAttention(
            768, num_heads, batch_first=True, dropout=dropout
        )
        self.norm       = nn.LayerNorm(768)

    def _encode_text(self, input_ids, attention_mask) -> torch.Tensor:
        out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        return out.last_hidden_state[:, 0, :]        # (B, 768)

    def _encode_image(self, pixel_values, has_image) -> torch.Tensor:
        B     = pixel_values.size(0)
        embed = torch.zeros(B, 768, device=pixel_values.device)
        if has_image.any():
            idx        = has_image.nonzero(as_tuple=True)[0]
            out        = self.vit(pixel_values=pixel_values[idx])
            embed[idx] = out.last_hidden_state[:, 0, :]
        return embed                                 # (B, 768)

    def forward(self, ctx: AgentContext) -> AgentContext:
        """
        Reads  : ctx.input_ids, ctx.attention_mask,
                 ctx.pixel_values, ctx.has_image,
                 ctx.evidence_embeds (B, K, 768)
        Writes : ctx.text_embed, ctx.verified_embed,
                 ctx.cross_modal_score
        """
        text_embed  = self._encode_text(
            ctx.input_ids, ctx.attention_mask
        )                                            # (B, 768)
        image_embed = self._encode_image(
            ctx.pixel_values, ctx.has_image
        )                                            # (B, 768)

        # Cross-modal alignment score (cosine similarity)
        cross_modal_score = F.cosine_similarity(
            text_embed, image_embed, dim=-1
        )                                            # (B,)

        # Multimodal fusion
        fused = self.mm_fusion(
            torch.cat([text_embed, image_embed], dim=-1)  # (B, 1536)
        )                                                  # (B, 768)

        # Cross-attention: fused claim → evidence
        attended, _ = self.cross_attn(
            query=fused.unsqueeze(1),                # (B, 1, 768)
            key=ctx.evidence_embeds,                 # (B, K, 768)
            value=ctx.evidence_embeds,
        )                                            # (B, 1, 768)

        verified_embed = self.norm(
            attended.squeeze(1) + fused             # residual
        )                                            # (B, 768)

        ctx.text_embed       = text_embed
        ctx.verified_embed   = verified_embed
        ctx.cross_modal_score = cross_modal_score
        return ctx


# ─────────────────────────────────────────────
# 4. AGENT 3 — GRAPH ANALYST
# ─────────────────────────────────────────────

class GraphAnalystAgent(nn.Module):
    """
    Propagation structure analyser using a 2-layer GAT.

    Responsibilities:
        - Run GAT over the claim propagation graph
        - Derive structural embedding (global mean pool)
        - Compute propagation risk score from graph topology
        - Expose per-node attention for interpretability
    """

    def __init__(self, in_dim: int = 768,
                 hidden_dim: int = 256, out_dim: int = 128,
                 heads1: int = 4, heads2: int = 2,
                 dropout: float = 0.3):
        super().__init__()
        self.gat1 = GATConv(in_dim, hidden_dim,
                            heads=heads1, dropout=dropout, concat=True)
        self.gat2 = GATConv(hidden_dim * heads1, out_dim,
                            heads=heads2, dropout=dropout, concat=False)
        self.proj = nn.Linear(out_dim, 768)
        self.norm = nn.LayerNorm(768)
        self.drop = nn.Dropout(dropout)

        # Propagation risk MLP: graph stats → scalar
        self.risk_head = nn.Sequential(
            nn.Linear(out_dim + 2, 32),   # +2 = (avg_degree, depth_proxy)
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def _graph_stats(self, edge_index: torch.Tensor,
                     num_nodes: int, batch: torch.Tensor,
                     B: int) -> torch.Tensor:
        """
        Compute per-graph [avg_degree, log_num_nodes] as topology features.
        Returns (B, 2).
        """
        stats = torch.zeros(B, 2, device=edge_index.device)
        for g in range(B):
            node_mask  = (batch == g)
            n          = node_mask.sum().float()
            src        = edge_index[0]
            edge_mask  = node_mask[src]
            e          = edge_mask.sum().float()
            avg_degree = e / (n + 1e-6)
            stats[g]   = torch.tensor(
                [avg_degree.item(), torch.log(n + 1).item()],
                device=edge_index.device
            )
        return stats

    def forward(self, ctx: AgentContext) -> AgentContext:
        """
        Reads  : ctx.graph (PyG Batch)
        Writes : ctx.structural_embed, ctx.node_attention,
                 ctx.propagation_risk
        """
        dev        = ctx.input_ids.device
        x          = ctx.graph.x.to(dev)
        edge_index = ctx.graph.edge_index.to(dev)
        batch      = ctx.graph.batch.to(dev)
        B          = ctx.input_ids.size(0)

        # GAT layers
        h1 = F.elu(self.gat1(x, edge_index))          # (total_N, 256*4)
        h1 = self.drop(h1)
        h2 = F.elu(self.gat2(h1, edge_index))         # (total_N, 128)

        # Graph-level embedding
        pooled        = global_mean_pool(h2, batch)    # (B, 128)
        struct_embed  = self.norm(self.proj(pooled))   # (B, 768)

        # Propagation risk
        stats         = self._graph_stats(edge_index, x.size(0), batch, B)
        risk_input    = torch.cat([pooled, stats], dim=-1)  # (B, 130)
        prop_risk     = self.risk_head(risk_input).squeeze(-1)   # (B,)

        ctx.structural_embed  = struct_embed
        ctx.node_attention    = h2             # per-node for explainability
        ctx.propagation_risk  = prop_risk
        return ctx


# ─────────────────────────────────────────────
# 5. AGENT 4 — REASONER
# ─────────────────────────────────────────────

class ReasonerAgent(nn.Module):
    """
    Deliberative reasoning agent that synthesises all evidence and
    structural signals into a unified reasoning embedding.

    Responsibilities:
        - Fuse verified (multimodal+RAG) and structural (graph) embeddings
        - Apply a self-attention transformer block for deep reasoning
        - Produce a reasoning trace (token/node salience)
    """

    def __init__(self, embed_dim: int = 768,
                 nhead: int = 8, ff_dim: int = 2048,
                 dropout: float = 0.3):
        super().__init__()

        # Project joint (verified + structural) to reasoning space
        self.joint_proj = nn.Sequential(
            nn.Linear(768 + 768, 768),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Transformer encoder block (single layer of self-attention)
        self.transformer_block = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,     # pre-norm for stability
        )

        # Salient feature gate:
        # learns which dimensions matter most for final reasoning
        self.salience_gate = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid(),
        )

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, ctx: AgentContext) -> AgentContext:
        """
        Reads  : ctx.verified_embed   (B, 768)
                 ctx.structural_embed (B, 768)
        Writes : ctx.reasoned_embed   (B, 768)
                 ctx.reasoning_trace  (dict)
        """
        # Fuse verified + structural
        joint = self.joint_proj(
            torch.cat([ctx.verified_embed,
                       ctx.structural_embed], dim=-1)     # (B, 1536)
        )                                                  # (B, 768)

        # Self-attention reasoning (treat B as sequence-of-1 for now)
        seq        = joint.unsqueeze(1)                    # (B, 1, 768)
        reasoned   = self.transformer_block(seq)           # (B, 1, 768)
        reasoned   = reasoned.squeeze(1)                   # (B, 768)

        # Salience gate
        gate       = self.salience_gate(reasoned)          # (B, 768)
        gated      = reasoned * gate                       # (B, 768)
        out        = self.norm(gated + joint)              # residual

        # Reasoning trace: top-10 most salient feature dimensions
        salience_vals, salience_idxs = gate.detach().mean(0).topk(10)
        reasoning_trace = {
            "top_salient_dims":   salience_idxs.cpu().tolist(),
            "top_salience_vals":  salience_vals.cpu().tolist(),
        }

        ctx.reasoned_embed   = out
        ctx.reasoning_trace  = reasoning_trace
        return ctx


# ─────────────────────────────────────────────
# 6. AGENT 5 — DECISION AGGREGATOR
# ─────────────────────────────────────────────

class DecisionAggregatorAgent(nn.Module):
    """
    Final verdict agent with dynamic learned gating over all agents.

    Responsibilities:
        - Compute per-agent contribution weights (soft gating network)
        - Weighted combination of verified, structural, reasoned embeds
        - MLP classification head
        - Temperature scaling for calibrated confidence
    """

    def __init__(self, embed_dim: int = 768,
                 hidden_dim: int = 256, num_classes: int = 2,
                 dropout: float = 0.3, temperature: float = 1.5):
        super().__init__()

        # Gating network: learns how much to trust each agent
        # Input  : concatenated [verified | structural | reasoned]  (2304)
        # Output : 3 softmax weights
        self.gate_net = nn.Sequential(
            nn.Linear(embed_dim * 3, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 3),
            nn.Softmax(dim=-1),
        )

        # Post-gating projection
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Classifier
        self.classifier = nn.Linear(hidden_dim, num_classes)

        # Learnable temperature for confidence calibration
        self.log_temp = nn.Parameter(
            torch.tensor(np.log(temperature), dtype=torch.float)
        )

    def forward(self, ctx: AgentContext) -> AgentContext:
        """
        Reads  : ctx.verified_embed   (B, 768)
                 ctx.structural_embed (B, 768)
                 ctx.reasoned_embed   (B, 768)
        Writes : ctx.logits (B, 2), ctx.agent_weights (B, 3)
        """
        v = ctx.verified_embed      # (B, 768)
        s = ctx.structural_embed    # (B, 768)
        r = ctx.reasoned_embed      # (B, 768)

        # Gating weights
        gate_input   = torch.cat([v, s, r], dim=-1)   # (B, 2304)
        agent_weights = self.gate_net(gate_input)      # (B, 3)

        w_v = agent_weights[:, 0:1]   # (B, 1)
        w_s = agent_weights[:, 1:2]
        w_r = agent_weights[:, 2:3]

        # Weighted combination
        aggregated = w_v * v + w_s * s + w_r * r      # (B, 768)

        # Classify
        hidden   = self.fusion(aggregated)             # (B, hidden_dim)
        logits   = self.classifier(hidden)             # (B, 2)

        # Temperature scaling
        temp     = self.log_temp.exp()                 # scalar ≥ 0
        logits   = logits / temp

        ctx.logits        = logits
        ctx.agent_weights = agent_weights
        return ctx


# ─────────────────────────────────────────────
# 7. MARGA  (full orchestrator)
# ─────────────────────────────────────────────

class MARGA(nn.Module):
    """
    Multi-Agent Retrieval-Augmented Graph Architecture.

    Orchestrates all 5 agents in sequence via a shared AgentContext.

    Forward pass order:
        Verifier (text+image encoding first, so Retriever has text_embed)
        → Retriever
        → Graph Analyst
        → Reasoner
        → Decision Aggregator
    """

    def __init__(self,
                 bert_model_name: str  = "bert-base-uncased",
                 vit_model_name:  str  = "google/vit-base-patch16-224",
                 corpus_passages: list = None,
                 tokenizer: BertTokenizer = None,
                 device: str = "cuda",
                 top_k:  int = 5,
                 gat_hidden: int = 256, gat_out: int = 128,
                 dropout: float = 0.3):
        super().__init__()

        # Shared encoders (loaded once, referenced by multiple agents)
        bert = BertModel.from_pretrained(bert_model_name).to(device)
        vit  = ViTModel.from_pretrained(vit_model_name).to(device)

        # ── Agent instantiation ───────────────────────────────────────
        self.verifier   = VerifierAgent(bert, vit, dropout=dropout)

        self.retriever  = RetrieverAgent(
            bert_model=bert,
            tokenizer=tokenizer,
            corpus_passages=corpus_passages or [],
            device=device,
            top_k=top_k,
        )

        self.graph_analyst = GraphAnalystAgent(
            in_dim=768, hidden_dim=gat_hidden,
            out_dim=gat_out, dropout=dropout
        )

        self.reasoner   = ReasonerAgent(dropout=dropout)

        self.aggregator = DecisionAggregatorAgent(dropout=dropout)

    def build_evidence_index(self):
        """Delegate FAISS index construction to the Retriever agent."""
        self.retriever.build_index()

    def forward(self,
                input_ids:      torch.Tensor,
                attention_mask: torch.Tensor,
                pixel_values:   torch.Tensor,
                has_image:      torch.Tensor,
                graph:          Batch,
                ) -> AgentContext:
        """
        Runs the full 5-agent pipeline.

        Returns the fully populated AgentContext
        (all agent outputs accessible for training + explainability).
        """
        ctx = AgentContext(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            has_image=has_image,
            graph=graph,
        )

        # ── Agent execution order ─────────────────────────────────────
        # 1. Verifier first — produces text_embed needed by Retriever
        # Step 1: get text embedding first (minimal encoding)
        ctx.text_embed = self.verifier._encode_text(
            ctx.input_ids, ctx.attention_mask
        )

        # Step 2: retrieve evidence
        ctx = self.retriever(ctx)

        # Step 3: full verifier (now has evidence)
        ctx = self.verifier(ctx)

        # 3. Graph Analyst — independent of RAG
        ctx = self.graph_analyst(ctx)

        # 4. Reasoner — synthesises verified + structural
        ctx = self.reasoner(ctx)

        # 5. Decision Aggregator — final verdict
        ctx = self.aggregator(ctx)

        return ctx


# ─────────────────────────────────────────────
# 8. DATASET  (reuses GraphRAG dataset + collate)
# ─────────────────────────────────────────────

class ClaimGraphBuilder:
    def __init__(self, bert_model, tokenizer, device, max_nodes=50, max_text_len=128):
        self.bert = bert_model
        self.tokenizer = tokenizer
        self.device = device
        self.max_nodes = max_nodes
        self.max_text_len = max_text_len

    @torch.no_grad()
    def _encode_node_texts(self, texts):
        enc = self.tokenizer(texts, max_length=self.max_text_len,
                             padding=True, truncation=True, return_tensors="pt")
        out = self.bert(input_ids=enc["input_ids"].to(self.device),
                        attention_mask=enc["attention_mask"].to(self.device))
        return out.last_hidden_state[:, 0, :].cpu()

    def build(self, graph_dict):
        nodes = graph_dict.get("nodes", [])[:self.max_nodes]
        edges = graph_dict.get("edges", [])
        if not nodes:
            nodes = [{"id": 0, "text": "unknown claim", "type": "root"}]
        texts = [n["text"] for n in nodes]
        x = self._encode_node_texts(texts)
        if edges:
            valid_ids = {n["id"] for n in nodes}
            edges = [e for e in edges if e["src"] in valid_ids and e["dst"] in valid_ids]
            id_to_idx = {n["id"]: i for i, n in enumerate(nodes)}
            src_list = [id_to_idx[e["src"]] for e in edges]
            dst_list = [id_to_idx[e["dst"]] for e in edges]
            weights  = [e.get("weight", 1.0) for e in edges]
            edge_index = torch.tensor([src_list, dst_list], dtype=torch.long)
            edge_attr  = torch.tensor(weights, dtype=torch.float).unsqueeze(1)
        else:
            edge_index = torch.zeros(2, 1, dtype=torch.long)
            edge_attr  = torch.ones(1, 1, dtype=torch.float)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

    @staticmethod
    def stub_graph(claim_text):
        return {
            "nodes": [
                {"id": 0, "text": claim_text, "type": "root"},
                {"id": 1, "text": f"Reply: {claim_text[:40]}", "type": "reply"},
                {"id": 2, "text": f"RT: {claim_text[:40]}",    "type": "retweet"},
                {"id": 3, "text": "This is false.",             "type": "reply"},
            ],
            "edges": [
                {"src": 0, "dst": 1, "weight": 1.0},
                {"src": 0, "dst": 2, "weight": 2.5},
                {"src": 0, "dst": 3, "weight": 1.0},
            ],
        }


class MARGADataset(Dataset):
    def __init__(self, dataframe, bert_tokenizer, vit_extractor,
                 graph_builder, max_length=128):
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

        enc = self.tokenizer(text, max_length=self.max_length,
                             padding="max_length", truncation=True,
                             return_tensors="pt")

        image_path = row.get("image_path", None)
        has_image  = False
        if pd.notna(image_path) and os.path.exists(str(image_path)):
            try:
                img          = Image.open(image_path).convert("RGB")
                vit_in       = self.vit_extractor(images=img, return_tensors="pt")
                pixel_values = vit_in["pixel_values"].squeeze(0)
                has_image    = True
            except Exception:
                pixel_values = torch.zeros(3, 224, 224)
        else:
            pixel_values = torch.zeros(3, 224, 224)

        graph_dict = row.get("graph", None)
        if not isinstance(graph_dict, dict):
            graph_dict = ClaimGraphBuilder.stub_graph(text)
        graph_data = self.graph_builder.build(graph_dict)

        return {
            "input_ids":      enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "pixel_values":   pixel_values,
            "has_image":      torch.tensor(has_image, dtype=torch.bool),
            "graph":          graph_data,
            "label":          torch.tensor(label, dtype=torch.long),
        }


def marga_collate_fn(batch):
    graphs = Batch.from_data_list([item.pop("graph") for item in batch])
    out    = {k: torch.stack([b[k] for b in batch]) for k in batch[0]}
    out["graph"] = graphs
    return out


# ─────────────────────────────────────────────
# 9. TRAINING LOOP
# ─────────────────────────────────────────────

class MARGATrainer:
    """
    Trains MARGA with:
        - Primary loss   : CrossEntropyLoss
        - Auxiliary loss : consistency_regulariser
                           (penalises text-image misalignment in real samples)
    """

    def __init__(self, model: MARGA,
                 train_loader: DataLoader,
                 val_loader:   DataLoader,
                 device:  str   = "cuda",
                 lr:      float = 2e-5,
                 weight_decay: float = 0.01,
                 epochs:  int   = 5,
                 lambda_reg: float = 0.1):

        self.model        = model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.device       = device
        self.epochs       = epochs
        self.lambda_reg   = lambda_reg

        self.ce_loss   = nn.CrossEntropyLoss()
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

    def _consistency_loss(self, ctx: AgentContext,
                          labels: torch.Tensor) -> torch.Tensor:
        """
        Penalise low text-image alignment for real (label=0) samples.
        Real news should have consistent text-image representation.
        """
        real_mask = (labels == 0)
        if real_mask.sum() == 0:
            return torch.tensor(0.0, device=self.device)
        scores     = ctx.cross_modal_score[real_mask]    # higher = more aligned
        return (1.0 - scores).mean()                     # minimise misalignment

    def _run_batch(self, batch):
        input_ids      = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        pixel_values   = batch["pixel_values"].to(self.device)
        has_image      = batch["has_image"].to(self.device)
        labels         = batch["label"].to(self.device)
        graph          = batch["graph"].to(self.device)

        ctx = self.model(input_ids, attention_mask,
                         pixel_values, has_image, graph)
        return ctx, labels

    def train_epoch(self):
        self.model.train()
        total_loss, all_preds, all_labels = 0.0, [], []

        for batch in self.train_loader:
            ctx, labels = self._run_batch(batch)

            loss_ce  = self.ce_loss(ctx.logits, labels)
            loss_reg = self._consistency_loss(ctx, labels)
            loss     = loss_ce + self.lambda_reg * loss_reg

            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            all_preds.extend(torch.argmax(ctx.logits, 1).cpu().numpy())
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
        all_agent_weights = []

        for batch in self.val_loader:
            t0          = time.perf_counter()
            ctx, labels = self._run_batch(batch)
            lat_ms      = (time.perf_counter() - t0) * 1000 / ctx.logits.size(0)
            latencies.append(lat_ms)

            all_preds.extend(torch.argmax(ctx.logits, 1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_agent_weights.append(ctx.agent_weights.cpu())

        avg_agent_weights = torch.cat(all_agent_weights).mean(0).tolist()

        return {
            "accuracy":        accuracy_score(all_labels, all_preds),
            "f1_macro":        f1_score(all_labels, all_preds, average="macro"),
            "precision_macro": precision_score(all_labels, all_preds, average="macro"),
            "recall_macro":    recall_score(all_labels, all_preds, average="macro"),
            "latency_ms":      float(np.mean(latencies)),
            "agent_w_verifier":  round(avg_agent_weights[0], 4),
            "agent_w_graph":     round(avg_agent_weights[1], 4),
            "agent_w_reasoner":  round(avg_agent_weights[2], 4),
        }

    def fit(self):
        print("=" * 60)
        print("  MODEL 4 — MARGA TRAINING")
        print("=" * 60)
        for epoch in range(1, self.epochs + 1):
            train_m = self.train_epoch()
            val_m   = self.evaluate()
            print(
                f"Epoch {epoch}/{self.epochs} | "
                f"Loss: {train_m['loss']:.4f} | "
                f"Train Acc: {train_m['accuracy']:.4f} | "
                f"Val F1: {val_m['f1_macro']:.4f} | "
                f"Val Acc: {val_m['accuracy']:.4f} | "
                f"Latency: {val_m['latency_ms']:.2f} ms/sample"
            )
            print(
                f"         Agent Weights → "
                f"Verifier: {val_m['agent_w_verifier']:.3f} | "
                f"Graph: {val_m['agent_w_graph']:.3f} | "
                f"Reasoner: {val_m['agent_w_reasoner']:.3f}"
            )
        return val_m


# ─────────────────────────────────────────────
# 10. INFERENCE  (full explanation chain)
# ─────────────────────────────────────────────

class MARGAInference:
    """
    Full MARGA inference with complete explanation chain.

    Returns:
        label              — binary verdict
        confidence         — calibrated probability
        evidence_passages  — RAG-retrieved support
        reasoning_trace    — top salient feature dimensions
        node_attention     — per-node GAT scores (propagation)
        propagation_risk   — topology-based risk scalar
        cross_modal_score  — text-image alignment
        agent_weights      — learned per-agent contribution
        latency_ms         — end-to-end latency

    Usage:
        engine = MARGAInference(model, bert_tokenizer, vit_extractor, device)
        result = engine.predict("5G spreads virus", image=None, graph_dict=None)
    """

    LABEL_MAP = {0: "Real", 1: "Fake"}

    def __init__(self, model: MARGA,
                 bert_tokenizer: BertTokenizer,
                 ViTImageProcessor: ViTImageProcessor.from_pretrained(VIT_MODEL),
                 graph_builder: ClaimGraphBuilder,
                 device: str = "cuda",
                 max_len: int = 128):
        self.model          = model.eval().to(device)
        self.bert_tokenizer = bert_tokenizer
        self.vit_extractor  = vit_extractor
        self.graph_builder  = graph_builder
        self.device         = device
        self.max_len        = max_len

    @torch.no_grad()
   

    def predict(self, claim: str,
            image: Optional[Image.Image] = None,
            graph_dict: Optional[dict]   = None) -> dict:

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
            has_image    = torch.tensor([True],  device=self.device)
        else:
            pixel_values = torch.zeros(1, 3, 224, 224, device=self.device)
            has_image    = torch.tensor([False], device=self.device)

        # ── Graph ──────────────────────────────────────────────────────
        if graph_dict is None:
            graph_dict = ClaimGraphBuilder.stub_graph(claim)
        graph_data       = self.graph_builder.build(graph_dict)
        graph_data.batch = torch.zeros(
            graph_data.num_nodes, dtype=torch.long, device=self.device
        )
        graph_batch = Batch.from_data_list([graph_data.to(self.device)])

        # ── Full pipeline ─────────────────────────────────────────────
        ctx = self.model(input_ids, attention_mask,
                         pixel_values, has_image, graph_batch)

        elapsed_ms = (time.perf_counter() - t0) * 1000

        # ── Decode outputs ────────────────────────────────────────────
        probs      = torch.softmax(ctx.logits, dim=1).squeeze(0)
        pred_class = torch.argmax(probs).item()
        confidence = probs[pred_class].item()

        return {
            "label":             self.LABEL_MAP[pred_class],
            "confidence":        round(confidence, 4),
            "evidence_passages": ctx.retrieved_passages[0]
                                 if ctx.retrieved_passages else [],
            "reasoning_trace":   ctx.reasoning_trace,
            "node_attention":    ctx.node_attention.norm(dim=-1).cpu().tolist(),
            "propagation_risk":  round(ctx.propagation_risk[0].item(), 4),
            "cross_modal_score": round(ctx.cross_modal_score[0].item(), 4),
            "agent_weights": {
                "verifier":  round(ctx.agent_weights[0, 0].item(), 4),
                "graph":     round(ctx.agent_weights[0, 1].item(), 4),
                "reasoner":  round(ctx.agent_weights[0, 2].item(), 4),
            },
            "latency_ms":        round(elapsed_ms, 2),
        }


# ─────────────────────────────────────────────
# 11. ENTRY POINT
# ─────────────────────────────────────────────

    def load_dataset(fakenewsnet_csv: str, twitter_csv: str) -> pd.DataFrame:
        frames = []

    for path, text_col in [(fakenewsnet_csv, "title"), (twitter_csv, "text")]:
        if os.path.exists(path):
            df = pd.read_csv(path)

            # ✅ REMOVE DUPLICATE COLUMNS
            df = df.loc[:, ~df.columns.duplicated()]

            # ✅ Rename text column
            if text_col in df.columns:
                df = df.rename(columns={text_col: "text"})

            # ✅ Ensure required columns exist
            for col in ["image_path", "graph"]:
                if col not in df.columns:
                    df[col] = None

            frames.append(df[["text", "label", "image_path", "graph"]])

    # ✅ If no files found
    if not frames:
        print("[WARNING] No dataset CSVs found — using synthetic stub.")
        return pd.DataFrame({
            "text": [
                "Government confirms new vaccine is 99% effective.",
                "Scientists say drinking vinegar cures cancer.",
                "UN releases annual climate change report.",
                "Aliens landed in New York says anonymous source.",
            ],
            "label": [0, 1, 0, 1],
            "image_path": [None] * 4,
            "graph": [None] * 4,
        })

    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(int)

    return df.sample(frac=1, random_state=42).reset_index(drop=True)

def main():
    DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"
    BERT_MODEL      = "bert-base-uncased"
    VIT_MODEL       = "google/vit-base-patch16-224"
    MAX_LEN         = 128
    BATCH_SIZE      = 8
    EPOCHS          = 5
    TOP_K           = 5
    LAMBDA_REG      = 0.1
    FAKENEWSNET_CSV = "data/fakenewsnet.csv"
    TWITTER_CSV     = "data/twitter_misinfo.csv"

    print(f"Device: {DEVICE}")

    bert_tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
    vit_extractor  = ViTFeatureExtractor.from_pretrained(VIT_MODEL)

    # Shared BERT for graph node encoding
    bert_shared = BertModel.from_pretrained(BERT_MODEL).to(DEVICE)

    # ── Dataset ───────────────────────────────────────────────────────
    # ─────────────────────────────────────────────
# DATA LOADING (FIXED)
# ─────────────────────────────────────────────

df = load_dataset(FAKENEWSNET_CSV, TWITTER_CSV)

# ✅ TEXT CLEANING
df["text"] = (
    df["text"]
    .astype(str)
    .str.lower()
    .str.strip()
    .str.replace(r"\s+", " ", regex=True)
)

# ✅ REMOVE LOW-QUALITY DATA
df = df[df["text"].str.len() > 30]

# ✅ REMOVE DUPLICATES
df = df.drop_duplicates(subset=["text"])


def load_dataset(fakenewsnet_csv: str, twitter_csv: str) -> pd.DataFrame:
    frames = []

    for path, text_col in [(fakenewsnet_csv, "title"), (twitter_csv, "text")]:
        if os.path.exists(path):
            df = pd.read_csv(path)

            # ✅ FIX: remove duplicate columns
            df = df.loc[:, ~df.columns.duplicated()]

            # rename text column
            if text_col in df.columns:
                df = df.rename(columns={text_col: "text"})

            # ensure required columns exist
            for col in ["image_path", "graph"]:
                if col not in df.columns:
                    df[col] = None

            frames.append(df[["text", "label", "image_path", "graph"]])

    if not frames:
        print("[WARNING] No dataset CSVs found — using synthetic stub.")
        return pd.DataFrame({
            "text": [
                "Government confirms new vaccine is 99% effective.",
                "Scientists say drinking vinegar cures cancer.",
                "UN releases annual climate change report.",
                "Aliens landed in New York says anonymous source.",
            ],
            "label": [0, 1, 0, 1],
            "image_path": [None] * 4,
            "graph": [None] * 4,
        })

    df = pd.concat(frames, ignore_index=True).dropna(subset=["text", "label"])
    df["label"] = df["label"].astype(int)

    return df.sample(frac=1, random_state=42).reset_index(drop=True)

    # ── MARGA model ───────────────────────────────────────────────────
    corpus_passages = train_df["text"].dropna().sample(
    min(2000, len(train_df)), random_state=42
).tolist()

    model = MARGA(
        bert_model_name=BERT_MODEL,
        vit_model_name=VIT_MODEL,
        corpus_passages=corpus_passages,
        tokenizer=bert_tokenizer,
        device=DEVICE,
        top_k=TOP_K,
    )
    model.build_evidence_index()

    # ── Train ─────────────────────────────────────────────────────────
    trainer = MARGATrainer(
        model, train_loader, val_loader,
        device=DEVICE, epochs=EPOCHS, lambda_reg=LAMBDA_REG
    )
    final_metrics = trainer.fit()

    print("\n── Final Validation Metrics ──────────────────────────────")
    for k, v in final_metrics.items():
        print(f"  {k:25s}: {v}")

    # ── Inference demo ────────────────────────────────────────────────
    engine = MARGAInference(
        model, bert_tokenizer, vit_extractor, graph_builder, DEVICE
    )
    print("\n── MARGA Inference Demo ──────────────────────────────────")
    test_claims = [
        "Scientists confirm 5G towers spread coronavirus.",
        "WHO publishes guidelines for pandemic preparedness.",
    ]
    for claim in test_claims:
        r = engine.predict(claim)
        print(f"\n  Claim              : {claim}")
        print(f"  Label              : {r['label']}  (conf={r['confidence']:.2%})")
        print(f"  Latency            : {r['latency_ms']} ms")
        print(f"  Propagation Risk   : {r['propagation_risk']}")
        print(f"  Cross-Modal Score  : {r['cross_modal_score']}")
        print(f"  Agent Weights      : {r['agent_weights']}")
        print(f"  Reasoning Trace    : top dims {r['reasoning_trace'].get('top_salient_dims', [])}")
        print(f"  Node Attention     : {[round(s,3) for s in r['node_attention']]}")
        print(f"  Evidence [1]       : {r['evidence_passages'][0][:100]}…"
              if r['evidence_passages'] else "  Evidence          : none")

    # ── Save ──────────────────────────────────────────────────────────
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/model4_marga.pt")
    print("\nCheckpoint saved → checkpoints/model4_marga.pt")


if __name__ == "__main__":
    main()