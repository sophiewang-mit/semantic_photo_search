# Semantic Photo Search

A semantic image retrieval system that allows users to search images using natural-language descriptions. This project combines CLIP embeddings, FAISS vector search, and a two-stage retrieval pipeline with reranking to return the most similar images to a text query.

## Features
* Natural-language image search using CLIP embeddings
* Fast vector retrieval with FAISS
* Two-stage retrieval pipeline:
  * Candidate generation using CLIP similarity
  * Learned reranking for refined ranking
* React frontend for image upload and search
* FastAPI backend for indexing and retrieval
* Evaluation framework using Recall@K, MRR, and nDCG
* Drag-and-drop image uploads

## Demo

Example queries:

* "a boat on water"
* "person playing instrument"
* "a dog on a mountain"
* "a person sleeping"

The system retrieves the top 6 semantically similar images, even when the exact words do not appear in filenames.

## Architecture

The system uses a two-stage retrieval pipeline:

```text
User Query
     │
     ▼
CLIP Text Encoder
     │
     ▼
FAISS Vector Search
     │
     ▼
Top-100 Candidates
     │
     ▼
Confidence Estimation
(Margin-Based Gating)
     │
     ├── High Confidence → Return CLIP Ranking
     │
     └── Low Confidence
              │
              ▼
      Logistic Reranker
              │
              ▼
      Final Ranked Results
```

### Stage 1: CLIP Retrieval

Images and text are embedded into a shared semantic space using CLIP. Candidate images are retrieved using cosine similarity and FAISS approximate nearest-neighbor search.

### Stage 2: Learned Reranking

A logistic regression reranker is trained on Flickr30k image-caption pairs using:

* Cosine similarity
* Embedding differences
* Elementwise embedding interactions

The reranker predicts the probability that a query-image pair is a semantic match.

### Confidence-Gated Inference

Experiments showed that always reranking degraded retrieval quality on Flickr30k. To address this, a confidence-based gating mechanism was implemented that only applies reranking when CLIP's top candidates are ambiguous.

This preserves strong CLIP performance while selectively applying additional computation when retrieval confidence is low.

## Evaluation

Retrieval quality was evaluated on held-out Flickr30k image-caption pairs using:

* Recall@1
* Recall@5
* Recall@10
* Mean Reciprocal Rank (MRR)
* nDCG@10

### Test Set Results

| Metric    | CLIP   | Always Rerank | Confidence-Gated |
| --------- | ------ | ------------- | ---------------- |
| Recall@1  | 90.4%  | 88.0%         | 90.4%            |
| Recall@5  | 98.8%  | 98.8%         | 98.8%            |
| Recall@10 | 100.0% | 100.0%        | 100.0%           |
| MRR       | 94.2%  | 93.1%         | 94.2%            |
| nDCG@10   | 99.7%  | 98.9%         | 99.7%            |

These results demonstrate that CLIP is already highly effective on Flickr30k retrieval, while the confidence-gated reranking framework provides a mechanism for selectively applying additional ranking models without degrading baseline performance.
