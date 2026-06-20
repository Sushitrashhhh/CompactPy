import time
import random
import matplotlib.pyplot as plt
from compactpy.memory import HierarchicalMemory
from compactpy.scoring import MemoryScoringEngine
from compactpy.compressors.attention import AttentionAwareCompressor
from compactpy.token_utils import count_tokens

def generate_mock_corpus(num_paragraphs):
    tech_facts = [
        "Mediscan AI leverages a high-performance FastAPI framework for backend processing.",
        "The custom OCR pipeline handles image pre-processing and text extraction rapidly.",
        "Deep learning models run inference on specialized GPU cluster arrays.",
        "We are actively monitoring API response latency across the network architecture."
    ]
    noise_facts = [
        "The weather in New Delhi fluctuates heavily between sunny and overcast conditions.",
        "A regular fitness regimen requires hitting the gym consistently six days a week.",
        "Open-source software projects rely heavily on clear automated documentation pipelines.",
        "Coffee consumption spikes among engineers during midnight deployment testing runs."
    ]
    
    corpus = []
    for _ in range(num_paragraphs):
        # Mix technical facts and noise
        corpus.append(random.choice(tech_facts))
        corpus.append(random.choice(noise_facts))
    return corpus

def run_performance_benchmark():
    print("==========================================================")
    print("         COMPACTPY ARCHITECTURE BENCHMARK RUNNER           ")
    print("==========================================================\n")

    attention_compressor = AttentionAwareCompressor()
    scoring_engine = MemoryScoringEngine()
    
    # Configuration arrays for scaling data sizes
    data_scales = [2, 5, 10, 15, 20, 30]
    token_counts = []
    reduction_percentages = []
    execution_times = []
    
    query = "What framework infrastructure powers the Mediscan AI backend?"
    budget = 40
    
    print("🚀 Simulating scaling context loads...")
    for scale in data_scales:
        memory_vault = HierarchicalMemory()
        raw_paragraphs = generate_mock_corpus(scale)
        full_text_pool = " ".join(raw_paragraphs)
        
        initial_tokens = count_tokens(full_text_pool)
        token_counts.append(initial_tokens)
        
        # Track execution time down to milliseconds
        start_time = time.perf_counter()
        
        # 1. Ingestion & Volatile Tier Routing
        for block in raw_paragraphs:
            importance = 0.85 if "FastAPI" in block or "OCR" in block else 0.3
            memory_vault.add_memory(block, importance=importance, utility=0.7)
            
        # 2. Lifecycle Scoring Simulation
        scoring_engine.process_lifecycle_cycle(memory_vault)
        active_pool = [m["text"] for m in memory_vault.working_memory]
        
        # 3. Dynamic Attention Budgeting Gating
        _, metrics = attention_compressor.compress_context_for_query(
            query=query,
            context_pool=active_pool,
            token_budget=budget
        )
        
        end_time = time.perf_counter()
        elapsed_ms = (end_time - start_time) * 1000
        
        reduction_percentages.append(metrics['reduction_percentage'])
        execution_times.append(elapsed_ms)
        
        print(f"   📊 Scale: {initial_tokens} Initial Tokens -> Reduced by {metrics['reduction_percentage']:.2f}% in {elapsed_ms:.2f}ms")

    # --- Plotting the Visual Performance Graph ---
    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Plot 1: Reduction Curve
    color = 'tab:blue'
    ax1.set_xlabel('Initial Prompt Token Footprint', fontweight='bold')
    ax1.set_ylabel('Token Space Reduction (%)', color=color, fontweight='bold')
    ax1.plot(token_counts, reduction_percentages, color=color, marker='o', linewidth=2.5, label='Reduction %')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.6)

    # Plot 2: Latency Curve (Shared X-axis)
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('Execution Latency (ms)', color=color, fontweight='bold')
    ax2.plot(token_counts, execution_times, color=color, marker='x', linestyle='--', linewidth=2, label='Latency (ms)')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('CompactPy V1.0.1: Efficiency Gains vs. Scaling Context Loads', fontsize=14, fontweight='bold', pad=15)
    fig.tight_layout()
    
    # Save chart straight to disk
    graph_filename = 'compactpy_benchmark_curve.png'
    plt.savefig(graph_filename, dpi=300)
    print(f"\n🎉 Benchmark complete! Performance curves saved visually to '{graph_filename}'")

if __name__ == "__main__":
    run_performance_benchmark()