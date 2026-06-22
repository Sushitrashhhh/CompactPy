import sys
import os

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__) + "/.."))

from compactpy.graph_memory.graph import GraphMemorySystem
from compactpy.graph_memory.linking import GraphMemoryLinker
from compactpy.graph_memory.algorithms import GraphTraversalEngine

def main():
    print("==========================================================")
    print("       COMPACTPY PHASE 3: MULTI-HOP PATH TRAVERSAL        ")
    print("==========================================================\n")
    
    gms = GraphMemorySystem()
    linker = GraphMemoryLinker(gms)
    traverser = GraphTraversalEngine(gms)
    
    # Inject an interconnected multi-hop story string
    deep_context_chain = (
        "Mediscan AI leverages a high-performance FastAPI framework backend setup. "
        "FastAPI powers container clusters easily. "
        "Container clusters deploy microservices safely."
    )
    
    print("📥 Ingesting relational context stream...")
    linker.ingest_unstructured_text(deep_context_chain)
    
    # 1. Look at immediate standard lookup (Depth 1)
    print("\n🔍 Standard Neighbors (1-Hop) for 'Mediscan AI':")
    for connection in gms.find_connected_concepts("Mediscan AI"):
        print(f"   • {connection}")
        
    # 2. Run your new multi-hop traversal algorithm (Depth 2)
    print("\n🕸️ Running Spreading Subgraph Traversal (2-Hop Explorer) for 'Mediscan AI':")
    subgraph_statements = traverser.traverse_subgraph_context("Mediscan AI", max_depth=2)
    
    if subgraph_statements:
        for statement in subgraph_statements:
            print(f"   • {statement}")
    else:
        print("   ❌ No deep context paths uncovered.")
    print("==========================================================")

if __name__ == "__main__":
    main()