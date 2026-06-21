from pathlib import Path
import sys
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score

# Gracefully handle the new nested /bin/ path resolution
try:
    from compactpy.memory import HierarchicalMemory, MemoryScoringEngine
    from compactpy.ml.ranker import MemoryRankingEngine
    from compactpy.compressors.attention import AttentionAwareCompressor
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from compactpy.memory import HierarchicalMemory, MemoryScoringEngine
    from compactpy.ml.ranker import MemoryRankingEngine
    from compactpy.compressors.attention import AttentionAwareCompressor

def run_ml_vs_heuristic_evaluation():
    print("==========================================================")
    print("         COMPACTPY PHASE 2: SYSTEM METRICS BENCH          ")
    print("==========================================================\n")

    # Ground Truth Dataset: (Context Chunk, Is_Actually_Relevant_To_Query)
    validation_dataset = [
        ("Mediscan AI leverages a high-performance FastAPI framework backend setup.", 1),
        ("The custom OCR pipeline handles image extraction workflows inside python.", 1),
        ("New Delhi is experiencing heavy overcast weather patterns today.", 0),
        ("A regular fitness regimen requires hitting the gym consistently six days a week.", 0),
        ("FastAPI instances can be deployed inside container clusters easily.", 1),
        ("Tomorrow's forecast predicts localized rainfall sweeps across Delhi.", 0)
    ]

    query = "What backend framework tools power the Mediscan AI pipeline?"
    ground_truth = [item[1] for item in validation_dataset]

    # --- 1. EVALUATE OLD HEURISTIC ROUTING ---
    memory_vault_h = HierarchicalMemory()
    scoring_engine_h = MemoryScoringEngine()
    attention_compressor = AttentionAwareCompressor()

    for text, _ in validation_dataset:
        # Simulating basic importance tagging
        imp = 0.9 if "FastAPI" in text or "OCR" in text else 0.3
        memory_vault_h.add_memory(text, importance=imp, utility=0.6)
    
    scoring_engine_h.process_lifecycle_cycle(memory_vault_h)
    active_pool_h = [m["text"] for m in memory_vault_h.working_memory]
    
    compressed_h, _ = attention_compressor.compress_context_for_query(
        query=query, context_pool=active_pool_h, token_budget=80
    )

    # --- 2. EVALUATE NEW PHASE 2 ML RANKER ROUTING ---
    ranker_engine = MemoryRankingEngine()
    ranker_engine.train_baseline_ranker() # Train the Random Forest
    
    # Structure data format for the ML Ranker input
    raw_memories = []
    for text, _ in validation_dataset:
        imp = 0.9 if "FastAPI" in text or "OCR" in text else 0.3
        raw_memories.append({"text": text, "importance": imp, "utility": 0.6, "recency": 1.0, "frequency": 1})
    
    # Predict scores using Random Forest feature engineering
    ml_scores = ranker_engine.predict_importance_scores(query, raw_memories)
    
    # Select items that clear a baseline threshold
    active_pool_ml = [raw_memories[i]["text"] for i in range(len(raw_memories)) if ml_scores[i] >= 0.4]
    
    compressed_ml, _ = attention_compressor.compress_context_for_query(
        query=query, context_pool=active_pool_ml, token_budget=80
    )

    # --- 3. CALCULATE STATISTICAL SCORES ---
    preds_h = [1 if item[0] in compressed_h else 0 for item in validation_dataset]
    preds_ml = [1 if item[0] in compressed_ml else 0 for item in validation_dataset]

    print("📊 Heuristic Formula Performance:")
    print(f"   • Precision : {precision_score(ground_truth, preds_h, zero_division=0)*100:.2f}%")
    print(f"   • Recall    : {recall_score(ground_truth, preds_h, zero_division=0)*100:.2f}%")
    print(f"   • F1-Score  : {f1_score(ground_truth, preds_h, zero_division=0)*100:.2f}%\n")

    print("🧠 Phase 2 Machine Learning Ranker Performance:")
    print(f"   • Precision : {precision_score(ground_truth, preds_ml, zero_division=0)*100:.2f}%")
    print(f"   • Recall    : {recall_score(ground_truth, preds_ml, zero_division=0)*100:.2f}%")
    print(f"   • F1-Score  : {f1_score(ground_truth, preds_ml, zero_division=0)*100:.2f}%")
    print("\n==========================================================")

if __name__ == "__main__":
    run_ml_vs_heuristic_evaluation()