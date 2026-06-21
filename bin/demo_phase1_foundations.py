import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from compactpy.core.chunk import chunk_text_by_words
from compactpy.token_utils import count_tokens, calculate_savings


def run_step1_verification():
    print("--- [Step 1] Verifying Token & Text Utilities ---")

    sample_story = (
        "FastAPI is used as the backend framework for Mediscan AI. "
        "The application integrates OCR engines to automate medicine detection workflows. "
        "Engineers chose FastAPI because of its native support for asynchronous programming, "
        "built-in validation parsing, and incredibly fast operational execution parameters."
    )

    print("Original Word Count:", len(sample_story.split()))
    chunks = chunk_text_by_words(sample_story, chunk_size=15, overlap=4)

    print(f"Split into {len(chunks)} overlapping chunks:")
    for idx, chunk in enumerate(chunks):
        print(f"  Chunk {idx + 1}: {chunk}")

    print("\n--- Testing Token Metrics Accounting Engine ---")
    mock_compressed = "FastAPI powers the Mediscan AI backend with OCR medicine detection."

    report = calculate_savings(sample_story, mock_compressed)
    print(f"Original Token Count: {report['original_tokens']}")
    print(f"Compressed Token Count: {report['compressed_tokens']}")
    print(f"Tokens Saved: {report['tokens_saved']}")
    print(f"Context Reduction: {report['reduction_percentage']}%")


if __name__ == "__main__":
    run_step1_verification()