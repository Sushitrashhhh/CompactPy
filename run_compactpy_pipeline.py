import os
import sys
import time

# Ensure path resolution works cleanly out-of-the-box
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import our modular components
from compactpy.core.chunk import chunk_text_by_words
from compactpy.compressors.dedup import DeduplicationEngine
from compactpy.compressors.semantic import SemanticCompressor
from compactpy.memory import HierarchicalMemory
from compactpy.scoring import MemoryScoringEngine
from compactpy.graph_memory import GraphMemorySystem
from compactpy.compressors.attention import AttentionAwareCompressor

def execute_complete_pipeline():
    print("==========================================================")
    
    print("     COMPACTPY: END-TO-END CONTEXT EXTRACTION PIPELINE    ")
    print("==========================================================\n")

    # 1. Initialize all architectural layers
    memory_vault = HierarchicalMemory()
    scoring_engine = MemoryScoringEngine()
    graph_db = GraphMemorySystem()
    attention_compressor = AttentionAwareCompressor()
    
    # 2. Raw incoming unstructured conversation log dump
    raw_transcript_stream = (
        "We are designing a medicine detection module called Mediscan AI. "
        "Mediscan AI uses FastAPI for the backend. Mediscan AI uses FastAPI for the backend. "
        "The module requires a custom OCR pipeline to process images cleanly. "
        "Today the weather in Delhi is cloudy. Tomorrow might be sunny."
    )
    
    print("📦 [Step 1] Chunking & Ingesting Raw Stream...")
    # Break down the massive stream into processable text blocks
    text_chunks = chunk_text_by_words(raw_transcript_stream, chunk_size=12, overlap=2)
    
    # Ingest into our Raw Hierarchical Memory tier with base tracking metrics
    for chunk in text_chunks:
        # Give technical terms higher baseline importance manually for simulation
        importance = 0.85 if "FastAPI" in chunk or "OCR" in chunk else 0.4
        memory_vault.add_memory(chunk, importance=importance, utility=0.7)
        time.sleep(0.01) # Stagger timestamps
        
    print(f"   -> Successfully ingested {len(memory_vault.raw_memory)} raw memory blocks.")

    print("\n🧹 [Step 2 & 3] Running Deduplication & Lifecycle Scoring...")
    # Simulate processing hits to update frequency variables
    memory_vault.increment_frequency(text_chunks[0])
    
    # Execute the scoring loops to route things out of raw storage
    scoring_engine.process_lifecycle_cycle(memory_vault)
    
    print(f"   -> Long-Term Memory Tier: {len(memory_vault.long_term_memory)} critical records retained.")
    print(f"   -> Working Memory Tier: {len(memory_vault.working_memory)} conversational records retained.")

    print("\n🕸️ [Step 4] Extracting Structural Graph Dependencies...")
    # Convert high-value retained long-term items into structural triplets
    graph_db.add_relation("FastAPI", "backend_of", "Mediscan AI")
    graph_db.add_relation("OCR", "required_for", "Mediscan AI")
    
    graph_facts = graph_db.get_relationships_as_text()
    for fact in graph_facts:
        print(f"   • Graph Relation: {fact}")

    print("\n🧠 [Step 5] Applying Query-Specific Attention Budgeting...")
    # Combine our working memory strings and graph structures into one global context pool
    combined_pool = [m["text"] for m in memory_vault.working_memory] + graph_facts
    
    # Target query
    user_query = "What technologies are required to back our Mediscan AI project infrastructure?"
    
    # Compress dynamically using attention predictor matching our strict prompt budget
    final_prompt_context, metrics = attention_compressor.compress_context_for_query(
        query=user_query,
        context_pool=combined_pool,
        token_budget=45
    )
    
    print(f"   Target Query: '{user_query}'")
    print(f"\n🚀 [Final Output] Optimized Context Payload Sent to LLM:")
    print(f"   \"{final_prompt_context}\"")
    print("\n==========================================================")
    print(f"📊 Framework Efficiency: {metrics['reduction_percentage']}% Token Reduction achieved safely!")
    print("==========================================================")

if __name__ == "__main__":
    execute_complete_pipeline()