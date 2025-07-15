# 🧠 Understanding Hybrid Search in RAG Systems with Pinecone

This repository captures my learning and experimentation with **Hybrid Search** for **Retrieval-Augmented Generation (RAG)** — focusing on **BM25 (sparse)** and **dense vector** retrieval using **Pinecone**.

> 🔍 I haven’t built a full RAG app yet — this is a learning project to explore how hybrid search works and how it could be applied in RAG pipelines.

---

## 💡 What I Learned

### ✅ What Is Hybrid Search?

Hybrid search combines:
- **Sparse Retrieval (BM25)**: Keyword-based, exact matches, interpretable.
- **Dense Retrieval (Embeddings)**: Semantic similarity using vectors.

By combining both, we get more **accurate**, **relevant**, and **robust** search — especially important in open-domain QA and LLM apps.

---

## 🔧 Tools I Explored

- `pinecone`: Vector database supporting hybrid (dense + sparse) search
- `pinecone-text`: BM25 encoder to generate sparse vectors
- `sentence-transformers`: To generate dense vector embeddings
- `Python`: To write test scripts and understand the API

---

## 🧪 Code Snippet: Index Creation in Pinecone

```python
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key="your-api-key")
index_name = "hybrid-search-example"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=384,  # Must match your embedding dimension
        metric="dotproduct",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
