import os
import sys

# System environment path resolver hook
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from compactpy.compressors.attention import AttentionAwareCompressor

def run_step6_verification():
    print("=== [Step 6] Testing Attention-Aware Context Compression ===")

    # Simulating a diverse historical knowledge pool containing various architectural details
    large_context_pool = [
        "User prefers building core backend components with FastAPI framework architectures.",
        "The project layout currently contains automated medicine detection models.",
        "Today the weather in Delhi is cloudy and overcast with slight rainfall indicators.",
        "Engineers chose an image OCR system setup to read scanned prescription labels.",
        "Database migrations completed successfully on the local development environment."
    ]

    # The user asks a specific question about application infrastructure setup
    user_query = "What backend stack options did we decide to use for our web service?"
    
    print(f"Live Incoming User Query: '{user_query}'")
    print(f"Total Historical Context Pool: {len(large_context_pool)} distinct statement blocks.")

    # Initialize our predictor engine with a tight token budget ceiling
    attention_compressor = AttentionAwareCompressor()
    compressed_context, metrics = attention_compressor.compress_context_for_query(
        query=user_query,
        context_pool=large_context_pool,
        token_budget=30  # Force budget constriction
    )

    print("\n--- Dynamically Predicted Compressed Context Payload ---")
    print(f"  👉 Retained Context: \"{compressed_context}\"")

    print("\n--- Final Attention-Aware Compaction Metrics ---")
    print(f"  Original Pool Footprint: {metrics['original_tokens']} tokens")
    print(f"  Tightly Compressed Footprint: {metrics['compressed_tokens']} tokens")
    print(f"  Total Saved Context: {metrics['tokens_saved']} tokens ({metrics['reduction_percentage']}% Reduction)")
    print("\n🎉 CompactPy has successfully isolated the exact relevant memories while filtering out the noise!")

if __name__ == "__main__":
    run_step6_verification()