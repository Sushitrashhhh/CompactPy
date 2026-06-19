import os
import sys

# Maintain path injection for standalone file safety
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from compactpy.compressors.dedup import DeduplicationEngine
from compactpy.compressors.semantic import SemanticCompressor

def run_step2_verification():
    print("=== [Step 2] Testing Core Compression Engines ===")

    # ----------------------------------------------------
    # Test 1: Deduplication Engine (Version 1)
    # ----------------------------------------------------
    print("\n--- Running Version 1: Exact Deduplication ---")
    v1_input = [
        "User likes Python",
        "User likes Python",
        "User likes Machine Learning",
        "User likes Python"
    ]
    
    dedup_engine = DeduplicationEngine()
    v1_output, v1_metrics = dedup_engine.compress(v1_input)
    
    print("Original items:", len(v1_input))
    print("Deduplicated output:", v1_output)
    print(f"Tokens Saved: {v1_metrics['tokens_saved']} ({v1_metrics['reduction_percentage']}% Reduction)")

    # ----------------------------------------------------
    # Test 2: Semantic Compressor (Version 2)
    # ----------------------------------------------------
    print("\n--- Running Version 2: Semantic Compression ---")
    v2_input = [
        "I love coding in Python.",
        "Python is my absolute favorite language.",
        "I really enjoy writing Python scripts.",
        "Today the weather in Delhi is cloudy."
    ]
    
    # 0.75 threshold means highly similar phrases will be compressed away
    semantic_engine = SemanticCompressor(threshold=0.75)
    v2_output, v2_metrics = semantic_engine.compress(v2_input)
    
    print("Original statements:")
    for text in v2_input:
        print(f"  - {text}")
        
    print("\nCompressed semantic statements:")
    for text in v2_output:
        print(f"  - {text}")
        
    print(f"\nTokens Saved: {v2_metrics['tokens_saved']} ({v2_metrics['reduction_percentage']}% Reduction)")

if __name__ == "__main__":
    run_step2_verification()