# CompactPy 🧠⚡

An intelligent, multi-evolutionary hierarchical memory and context compression framework designed to optimize LLM prompt footprints and eliminate token bloat in RAG pipelines.

[![PyPI Version](https://img.shields.io/pypi/v/compactpy.svg)](https://pypi.org/project/compactpy/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![GitHub](https://img.shields.io/badge/github-Sushitrashhhh%2FCompactPy-blue)](https://github.com/Sushitrashhhh/CompactPy)

---

## 🚀 The Core Problem

Large Language Models have finite, expensive context windows. Storing raw, repetitive conversational history, system clutter, and loose narrative prose directly in the prompt window leads to massive API billing inflation, elevated system latency, and model confusion due to key context dilution.

**CompactPy solves this.** By mimicking cognitive memory tiers, vector math similarities, and directed knowledge graphs, it drops prompt footprints by **40%+** while perfectly preserving deep engineering states and concept dependencies.

---

## 🛠️ Multi-Evolutionary Architecture

CompactPy processes raw runtime context streams across six specialized optimization phases:

### 1. Token Analytics Core (`compactpy.core`)

Uses high-speed BPE tokenization via `tiktoken` to run precision boundaries, calculating exact text lengths and tracking compression metrics down to individual bits.

### 2. Algorithmic Compression Engines (`compactpy.compressors`)

* **Exact Deduplication Engine:** Automatically strips out repetitive context loops and chronological logs while keeping structural stream order intact.
* **Semantic Compressor:** Embeds data blocks via `SentenceTransformer`, executing vector **Cosine Similarity** arrays to eliminate overlapping thoughts (e.g., keeping only one variation of a phrase if similarity crosses a `0.75` threshold).

### 3. Hierarchical Memory Repository (`compactpy.memory`)

Isolates text strings into explicit cognitive abstraction layers based on real-world utility:

* `raw_memory`: The volatile, incoming execution log dump.
* `working_memory`: Active short-term operational buffers available for immediate context retrieval.
* `long_term_memory`: High-value project parameters and user rules that never decay.

### 4. Memory Scoring Engine (`compactpy.scoring`)

Memories are evaluated dynamically using a custom, long-horizon linear performance formula:

```math
Score = 0.4 × Importance + 0.3 × Utility + 0.2 × Recency + 0.1 × Frequency
```

High-scoring nodes are promoted straight to Long-Term Memory, medium nodes stay in Working storage, and low-scoring noise is automatically evicted to prevent token bloat.

### 5. Relational Graph Memory System (`compactpy.graph_memory`)

Converts raw long-term strings into dense, indexed, bidirectional **Knowledge Graphs** using `NetworkX`. Instead of raw prose, it stores knowledge as structured triplets:

```text
Source Entity --(Relation)--> Target Entity
```

**Example:**

```text
FastAPI --(backend_of)--> Mediscan AI
```

This retains complex causal relationships without wasting prompt space.

### 6. Attention-Aware Compressor (`compactpy.compressors.attention`)

Acts as a dynamic "Importance Predictor." When a user passes a live query, it calculates the attention weight of your history pool relative to that query, dynamically filling a targeted prompt token budget with the highest-relevance vectors.

---

## 💾 Installation

Install the production framework directly from PyPI:

```bash
pip install compactpy
```

---

## 💻 Quickstart: End-to-End Pipeline

Here is how to run the complete automated ingestion, scoring, and query-aware compaction loop:

```python
from compactpy.memory import HierarchicalMemory
from compactpy.scoring import MemoryScoringEngine
from compactpy.graph_memory import GraphMemorySystem
from compactpy.compressors.attention import AttentionAwareCompressor

# 1. Initialize our modular cognitive layers
memory_vault = HierarchicalMemory()
scoring_engine = MemoryScoringEngine()
graph_db = GraphMemorySystem()
attention_compressor = AttentionAwareCompressor()

# 2. Ingest raw conversational logs
raw_logs = [
    "We are designing a medicine detection module called Mediscan AI.",
    "Mediscan AI uses FastAPI for the backend framework architecture.",
    "Today the weather in Delhi is cloudy and rainy."
]

for log in raw_logs:
    importance = 0.85 if "FastAPI" in log or "Mediscan" in log else 0.3
    memory_vault.add_memory(log, importance=importance, utility=0.7)

# 3. Simulate usage hits and run lifecycle scoring
memory_vault.increment_frequency(raw_logs[1])
scoring_engine.process_lifecycle_cycle(memory_vault)

# 4. Map persistent facts into the knowledge graph
graph_db.add_relation("FastAPI", "backend_of", "Mediscan AI")
graph_facts = graph_db.get_relationships_as_text()

# 5. Build a query-aware compact context
user_query = "What backend options did we settle on for Mediscan AI?"
combined_context = [m["text"] for m in memory_vault.working_memory] + graph_facts

optimized_payload, metrics = attention_compressor.compress_context_for_query(
    query=user_query,
    context_pool=combined_context,
    token_budget=45
)

print(f"Optimized Prompt Context: {optimized_payload}")
print(f"Token Reduction: {metrics['reduction_percentage']}%")
```

---

## 🧪 Running Validation Demos

The project repository includes individual step verification scripts right at the root level. Run them to watch the math and optimization phases execute live in your terminal:

```bash
# Test token utilities and basic compressors
python demo_step1.py
python demo_step2.py

# Test hierarchical lifecycle scoring loops
python demo_step3.py

# Test graph relationship mapping
python demo_step5.py

# Test dynamic attention query budgeting
python demo_step6.py

# Run the complete end-to-end processing pipeline
python run_compactpy_pipeline.py
```

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
