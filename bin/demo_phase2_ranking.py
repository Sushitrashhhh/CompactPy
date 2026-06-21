import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from compactpy.ml.ranker import MemoryRankingEngine


def test_machine_learning_ranker():
    print("==========================================================")
    print("         COMPACTPY PHASE 2: ML RANKING SYSTEM VERIFIER     ")
    print("==========================================================\n")

    active_memories = [
        {"text": "Mediscan AI relies on FastAPI for high-throughput routing.", "importance": 0.9, "utility": 0.8, "recency": 0.9, "frequency": 4},
        {"text": "The weather outside right now in Delhi is warm and clear.", "importance": 0.2, "utility": 0.1, "recency": 0.8, "frequency": 1},
        {"text": "We configured an automated custom OCR pipeline for images.", "importance": 0.8, "utility": 0.7, "recency": 0.5, "frequency": 3},
        {"text": "I went to the local fitness gym early this morning.", "importance": 0.4, "utility": 0.3, "recency": 0.9, "frequency": 1}
    ]

    query = "What framework options handle backend requests for Mediscan AI?"

    print("📦 Loading regression model weights...")
    engine = MemoryRankingEngine()

    print("⚙️ Training model dynamically on baseline parameter profiles...")
    engine.train_baseline_ranker()

    print("⚙️ Scoring tracking metrics against active session search query...")
    predicted_scores = engine.predict_importance_scores(query, active_memories)

    print("\n🚀 ML Predictive Engine Value Output:")
    for idx, score in enumerate(predicted_scores):
        print(f"   • Score: {score:.4f} | \"{active_memories[idx]['text']}\"")


if __name__ == "__main__":
    test_machine_learning_ranker()