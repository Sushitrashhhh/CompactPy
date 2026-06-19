import time

class HierarchicalMemory:
    """
    Version 3: Hierarchical Memory Tier Architecture.
    Manages text data streams separated into explicit cognitive abstraction layers[cite: 69, 70, 71].
    """
    def __init__(self):
        # Initialize the lists exactly inside the constructor
        self.raw_memory = []       # Ephemeral input stream log [cite: 71]
        self.working_memory = []   # Active short-term operational context [cite: 71]
        self.summary_memory = []   # Abstracted/Summarized context blocks [cite: 71]
        self.long_term_memory = [] # Consolidated permanent values/rules [cite: 71]

    def add_memory(self, text: str, importance: float = 0.5, utility: float = 0.5, metadata: dict = None):
        """
        Deposits a new raw memory node into the system with initial tracking metrics.
        """
        memory_node = {
            "text": text,
            "timestamp": time.time(), 
            "frequency": 1,           
            "importance": importance, 
            "utility": utility,       
            "metadata": metadata or {}
        }
        self.raw_memory.append(memory_node)

    def increment_frequency(self, text_content: str):
        """
        Simulates memory usage by locating an existing entry and incrementing its frequency tracking score.
        """
        for list_tier in [self.raw_memory, self.working_memory, self.long_term_memory]:
            for node in list_tier:
                if node["text"] == text_content:
                    node["frequency"] += 1
                    return