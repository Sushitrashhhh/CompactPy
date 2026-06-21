import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from compactpy.ml.clustering import MemoryClusteringEngine


def test_machine_learning_clustering():
    print("==========================================================")
    print("       COMPACTPY PHASE 2: CLUSTERING SYSTEM VERIFIER      ")
    print("==========================================================\n")

    texts = [
        "Mediscan AI relies on FastAPI for high-throughput routing.",
        "The weather outside right now in Delhi is warm and clear.",
        "We configured an automated custom OCR pipeline for images.",
        "I went to the local fitness gym early this morning."
    ]

    engine = MemoryClusteringEngine()
    clusters = engine.cluster_with_kmeans(texts, n_clusters=2)

    print("🚀 KMeans cluster output:")
    for label, items in clusters.items():
        print(f"   Cluster {label}: {items}")


if __name__ == "__main__":
    test_machine_learning_clustering()