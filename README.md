# MARGA: Multi-Agent Retrieval-Augmented Graph Architecture

## Models
- Model 1: Baseline (BERT)
- Model 2: RAG-based (BERT + FAISS)
- Model 3: Graph-RAG (BERT + FAISS + GAT)
- Model 4: Full MARGA (multi-agent)

## Datasets
- FakeNewsNet (PolitiFact)
- Twitter15/16 (for graph models)

⚠️ Datasets are not included due to size.

## Setup
```bash
pip install torch transformers faiss-cpu pandas scikit-learn
