import os
import sys

# Standard runtime system environment path resolver hook
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from compactpy.graph_memory import GraphMemorySystem
from compactpy.token_utils import calculate_savings

def run_step5_verification():
    print("=== [Step 5] Testing Graph Memory System ===")
    
    # Initialize our network memory system
    graph_memory = GraphMemorySystem()

    # 1. Simulate feeding project architecture context sentences into the graph
    # Sentence A: "FastAPI is used as the backend framework for Mediscan AI."
    graph_memory.add_relation("FastAPI", "backend_of", "Mediscan AI")
    
    # Sentence B: "The application integrates OCR engines to automate medicine detection workflows."
    graph_memory.add_relation("OCR", "required_for", "Mediscan AI")
    graph_memory.add_relation("Mediscan AI", "performs", "Medicine Detection")

    print("\n--- [1] Displaying Structural Knowledge Graph Triplets ---")
    graph_statements = graph_memory.get_relationships_as_text()
    for statement in graph_statements:
        print(f"  Graph Edge: {statement}")

    print("\n--- [2] Running Target Concept Node Queries ---")
    query_node = "Mediscan AI"
    print(f"Retrieving active sub-network paths connected directly to '{query_node}':")
    related_paths = graph_memory.find_connected_concepts(query_node)
    for path in related_paths:
        print(f"  👉 Found Connection: {path}")

    print("\n--- [3] Calculating Context Size Reduction Benefits ---")
    # Compare raw descriptive prose against our dense structural string mappings
    raw_prose = (
        "We are building an application called Mediscan AI. For the backend setup of Mediscan AI, "
        "we are utilizing the FastAPI framework. Additionally, an OCR system is strictly required for "
        "Mediscan AI so that the application can successfully run medicine detection pipelines."
    )
    compressed_graph_text = " ".join(graph_statements)

    report = calculate_savings(raw_prose, compressed_graph_text)
    print(f"Raw Narrative Prompt Footprint: {report['original_tokens']} tokens")
    # Graph layout yields significant token minimization benefits
    print(f"Graph Structural Frame Footprint: {report['compressed_tokens']} tokens")
    print(f"Total Structural Savings: {report['tokens_saved']} tokens ({report['reduction_percentage']}% Reduction)")

if __name__ == "__main__":
    run_step5_verification()