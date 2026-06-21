from pathlib import Path
import sys

# Because this file is now nested inside bin/, we need to step out exactly two levels
# to reach the core COMPACTPY root workspace.
try:
    from compactpy.retrieval import BM25RetrievalEngine
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from compactpy.retrieval import BM25RetrievalEngine
def test_sparse_information_retrieval():
    print("==========================================================")
    print("         COMPACTPY PHASE 2: BM25 IR SYSTEM VERIFIER       ")
    print("==========================================================\n")

    corpus = [
        "Mediscan AI leverages a high-performance FastAPI framework backend setup.",
        "A regular fitness regimen requires hitting the gym consistently six days a week.",
        "The custom OCR pipeline handles image extraction workflows inside python.",
        "New Delhi is experiencing heavy overcast weather patterns today.",
        "Error token context exception thrown inside user_id_v2 initialization loop."
    ]

    # Explicit query looking for a highly specific technical phrase match
    query = "user_id_v2 exception token error"

    engine = BM25RetrievalEngine()
    print("⚙️ Analyzing vocabulary arrays and generating IDF values...")
    engine.fit(corpus)

    print(f"🔎 Executing sparse BM25 query lookup for: '{query}'")
    results = engine.search(query, corpus, top_n=2)

    print("\n🚀 Top Ranked Sparse Matching Context Results:")
    for text, score in results:
        print(f"   • [Score: {score:.4f}] -> \"{text}\"")

if __name__ == "__main__":
    test_sparse_information_retrieval()