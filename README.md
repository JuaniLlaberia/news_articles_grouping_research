# News Articles Grouping Research

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Paper](https://img.shields.io/badge/Paper-OpenReview-red.svg)](https://openreview.net/forum?id=b5XymdQ6Bj&referrer=%5Bthe%20profile%20of%20Juan%20Ignacio%20Llaberia%5D(%2Fprofile%3Fid%3D~Juan_Ignacio_Llaberia1))
[![NLP](https://img.shields.io/badge/Task-NLP-purple.svg)](https://en.wikipedia.org/wiki/Natural_language_processing)
[![Model](https://img.shields.io/badge/HuggingFace-cross--encoder-orange.svg)](https://huggingface.co/Juanillaberia/articles-pairs-event-detection)

This repository contains the **complete experimental pipeline and notebooks** for research on **automatic news article grouping by real-world events**.

---

## Overview

Grouping news articles by topic is straightforward. Grouping them by **specific real-world events** is not.

Consider two articles about the same public figure:

- *The Argentine president approves a set of laws preventing X, Y, and Z.*
- *The Argentine president vetoes a tax reform proposal.*

Both mention the same entity, but they describe **different events** and should belong to **different clusters**. A purely semantic approach will often fail here — the articles are topically similar but event-distinct.

This project tests whether combining multiple signals — semantic similarity, entity disambiguation, and cross-encoder pair classification — can produce more accurate and granular event-level groupings than a semantic baseline alone.

---

## Contributions

This work:

- Demonstrates the limitations of semantic-only clustering (HDBSCAN) for event detection
- Proposes a **multi-signal pipeline** combining KNN retrieval, cross-encoder reranking, and entity-based Jaccard similarity
- Fine-tunes and evaluates a **custom cross-encoder** (`Juanillaberia/articles-pairs-event-detection`) for same-event pair classification
- Conducts a **training data ablation** showing that class balance directly controls the precision/recall tradeoff in downstream clustering
- Reduces noise article rate from **38.7% → ~10%** compared to the baseline
- Provides **fully reproducible notebooks** with checkpointing for long-running GPU stages

---

## Dataset

| Property | Value |
|---|---|
| Articles (full) | ~163,753 |
| Articles (experiment subset) | ~170,000 (2,500 events, `random.seed(42)`) |
| Languages | English, Spanish |
| Labels | Ground-truth `event_id` per article |
| Platform | Google Colab Pro (NVIDIA A100, 40GB VRAM) |

The full dataset was subsampled to 2,500 events for practical iteration speed. The sampling decision is documented and justified inline. All prior checkpoints from the full dataset are preserved.

---

## Methods

### Baseline

| Component | Details |
|---|---|
| Embedder | `intfloat/multilingual-e5-large` (512-token head truncation) |
| Clustering | HDBSCAN |

### Main Pipeline

| Stage | Component | Details |
|---|---|---|
| 1 | Entity Extraction | `Jean-Baptiste/roberta-large-ner-english` via spaCy sentencizer |
| 2 | Entity Disambiguation | `facebook/mgenre-wiki` with `[START]`/`[END]` tagging + lookup cache |
| 3 | Entity Normalization | Wikipedia title normalization + deduplication |
| 4 | Semantic Retrieval | `intfloat/multilingual-e5-large` + FAISS KNN (top-100 neighbors) |
| 5 | Pair Classification | Fine-tuned ModernBERT cross-encoder (`Juanillaberia/articles-pairs-event-detection`) |
| 6 | Entity Similarity | Jaccard similarity over normalized canonical entity sets |
| 7 | Score Fusion | Weighted sum: `0.85 × cross-encoder + 0.15 × Jaccard` |
| 8 | Graph Clustering | Leiden community detection (resolution = 200, threshold = 0.5) |

---

## Results

### Baseline vs. Main Pipeline

| Metric | Baseline (HDBSCAN) | Main Pipeline |
|---|---|---|
| Homogeneity | ~0.98 | 0.9474 |
| Completeness | ~0.75 | 0.8589 |
| V-Measure | — | **0.9010** |
| Avg. Intra-cluster Cosine Similarity | ~0.88 | **0.9573** |
| Article Coverage | 61.3% | **89.4%** |
| Noise Rate | ~38.7% | ~10% |

The main pipeline trades a small amount of homogeneity for substantial gains in completeness, V-Measure, semantic coherence, and — most critically — coverage.

### Cross-Encoder Training Ablation

| Metric | v1 (imbalanced, 20k pairs) | v2 (balanced, in-domain) |
|---|---|---|
| Homogeneity | 0.9474 | 0.8776 |
| Completeness | 0.8589 | 0.9306 |
| V-Measure | 0.9010 | 0.9034 |
| Intra-cluster Similarity | 0.9576 | 0.8843 |
| Article Coverage | ~90% | ~100% |

Training data balance directly controls the precision/recall tradeoff in downstream clustering, independently of graph construction and community detection.

### ANN vs. KNN Evaluation

| Method | Runtime | Recall@100 |
|---|---|---|
| Exact KNN (FAISS) | 3 seconds | 1.0 (reference) |
| ANN (FAISS IVF) | 16 seconds | 0.9654 |

KNN was selected as the retrieval method. ANN is noted as a preferable option if the dataset grows significantly.

### Graph Statistics (Main Pipeline)

| Statistic | Value |
|---|---|
| Article coverage | 146,402 / 163,753 (89.4%) |
| Unique clusters | 6,538 |
| Graph modularity | 0.9306 |

---

## Key Findings

- **Semantic similarity alone is insufficient** for event-level grouping. HDBSCAN merges topically similar but event-distinct articles and marks ~39% of articles as noise.
- **Multi-signal fusion** (cross-encoder + entity Jaccard) yields substantially better completeness and coverage with only a minor drop in purity.
- **Training data balance** is a direct lever on clustering behavior: imbalanced training favors purity; balanced training favors coverage.
- **Entity Jaccard should be weighted conservatively** — true same-event pairs often share few entities (mean Jaccard ≈ 0.28), so over-weighting entity signals penalizes valid pairs.

---

## Evaluation Metrics

- Homogeneity
- Completeness
- V-Measure
- Average Intra-cluster Cosine Similarity
- Article Coverage / Noise Rate
- Graph Modularity

---

## Repository Structure

| Notebook | Description |
|---|---|
| `main.ipynb` | Full multi-signal pipeline: NER → NEL → KNN → cross-encoder → Jaccard → graph clustering |
| `baseline.ipynb` | Semantic baseline: `multilingual-e5-large` embeddings + HDBSCAN |
| `cross_encoder_classifier.ipynb` | Cross-encoder fine-tuning and evaluation |
| `articles_events.ipynb` | Dataset exploration and event analysis |

---

## Computational Requirements

| Stage | Runtime (A100) |
|---|---|
| Entity pipeline (NER + NEL + normalization) | ~3h 30m |
| Embedding generation (163,753 articles) | ~44 minutes |
| KNN retrieval (top-100, full dataset) | 3 seconds |
| Cross-encoder inference (16.2M pairs) | ~1h 30m |
| **Total** | **~6 hours** |

All stages use HuggingFace `datasets` with `save_to_disk()` checkpointing. GPU stages use `del model` + `torch.cuda.empty_cache()` between steps to manage VRAM.

---

## Future Work

- **Weighted Jaccard similarity** — assign higher importance to entity types (e.g., events and locations over common persons)
- **Alternative embedding models** — evaluate other multilingual encoders for the retrieval stage
- **Temporal features** — incorporate publication date as an additional signal (preliminary results show ~3% improvement)
- **ANN at scale** — revisit approximate retrieval for full-dataset runs
- **End-to-end multilingual evaluation** — extend evaluation beyond English/Spanish

---

## Citation

```bibtex
@misc{llaberia2026newsarticlesgrouping,
  title={News Articles Grouping Research},
  author={Juan Ignacio Llaberia},
  year={2026},
  url={https://github.com/JuaniLlaberia/news_articles_grouping_research}
}
```

---

## License

Apache 2.0
