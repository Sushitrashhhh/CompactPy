import os
import sys
import time

# Dynamic standalone script system path resolver hook
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

from compactpy.memory import HierarchicalMemory
from compactpy.scoring import MemoryScoringEngine

def run_step3_verification():
    print("=== [Step 3 & 4] Testing Hierarchical Memory and Scoring Engine ===")
    
    # Initialize our tiered memory core and scoring lifecycle managers
    memory_manager = HierarchicalMemory()
    scoring_engine = MemoryScoringEngine()

    print("\n--- [1] Adding Diverse Context Logs into Raw Memory System ---")
    
    # Item A: High importance developer preference entry
    memory_manager.add_memory(
        text="User prefers building core backend components with FastAPI.",
        importance=0.9, 
        utility=0.8
    )
    time.sleep(0.1) # Stagger timestamps to let the recency metrics scale accurately

    # Item B: Medium value project situational details
    memory_manager.add_memory(
        text="Current development focus is an AI module named Mediscan AI.",
        importance=0.6, 
        utility=0.7
    )
    time.sleep(0.1)

    # Item C: Low importance runtime debug tracking information
    memory_manager.add_memory(
        text="Local standard output log statement: build connection compiled successfully.",
        importance=0.1, 
        utility=0.2
    )

    print(f"Total Raw Logs Pending Evaluation: {len(memory_manager.raw_memory)}")

    print("\n--- [2] Simulating Frequent Memory Usage Access Hits ---")
    # Simulate the user mentioning the FastAPI preference again, incrementing its frequency parameter
    memory_manager.increment_frequency("User prefers building core backend components with FastAPI.")
    print("  -> Log entry regarding 'FastAPI' accessed again! Frequency counter updated.")

    print("\n--- [3] Triggering Automated Compaction and Lifecycle Routing ---")
    scoring_engine.process_lifecycle_cycle(memory_manager)

    print("\n--- Evaluation Tier Summary Outputs ---")
    
    print("\n📦 LONG-TERM MEMORY TIER (High Score >= 0.70):")
    for m in memory_manager.long_term_memory:
        print(f"  ⭐ [Score: {m['lifecycle_score']}] - {m['text']}")

    print("\n⚙️ WORKING MEMORY TIER (Medium Score 0.35 - 0.69):")
    for m in memory_manager.working_memory:
        print(f"  🔄 [Score: {m['lifecycle_score']}] - {m['text']}")

    # System clutter logs whose total combined metric falls below 0.35 are automatically discarded
    print("\n🗑️ EVICTION REPORT:")
    total_retained = len(memory_manager.long_term_memory) + len(memory_manager.working_memory)
    evicted_count = len(memory_manager.raw_memory) - total_retained
    print(f"  Successfully evicted {evicted_count} low-value context logs, preventing token footprint bloat!")

if __name__ == "__main__":
    run_step3_verification()