import time 

class MemoryScoringEngine:
    '''
    ver4: memory scoring engine that evaluates memory items based on importance, utility, frequency, and recency. 
    '''

    def __init__(self, w_imp:float=0.4, w_util:float=0.3, w_freq:float=0.2, w_rec:float=0.2):
        self.w_imp = w_imp
        self.w_util = w_util
        self.w_freq = w_freq
        self.w_rec = w_rec

    def calculate_score(self, importance: float, utility: float, frequency: int, recency: float) -> float:
        """
        Calculate the memory score based on importance, utility, frequency, and recency.

        Args:
            importance (float): Importance score of the memory (0.0 to 1.0).
            utility (float): Utility score of the memory (0.0 to 1.0).
            frequency (int): Frequency count of how many times the memory has been accessed.
            timestamp (float): The time when the memory was last accessed.
            """
        return (self.w_imp * importance) + (self.w_util * utility) + (self.w_freq * frequency) + (self.w_rec * recency)
    
    def process_lifecycle_cycle(self, memory_system):
        """
        Process the lifecycle cycle for memory items, including scoring and tier management.

        Args:
            memory_system (HeirarchicalMemory): The memory system containing memory items to be processed.
        """
        all_memories = memory_system.raw_memory
        if not all_memories:
            return

        current_time = time.time()
        timestamps = [m["timestamp"] for m in all_memories]
        max_t, min_t = max(timestamps), min(timestamps)
        t_range = max_t - min_t if max_t != min_t else 1.0

        # Create temporary lists to hold current cycle route allocations
        promoted_long_term = []
        kept_working = []

        for m in all_memories:
            # 1. Compute dynamic normalized recency (1.0 = newest, 0.0 = oldest)
            normalized_recency = (m["timestamp"] - min_t) / t_range
            
            # 2. Normalize frequency baseline safely
            normalized_frequency = min(m["frequency"] / 5.0, 1.0)

            # 3. Compute final lifecycle performance score
            final_score = self.calculate_score(
                importance=m["importance"],
                utility=m["utility"],
                recency=normalized_recency,
                frequency=normalized_frequency
            )
            m["lifecycle_score"] = round(final_score, 3)

            # 4. Version 4 Lifecycle Routing Rules
            if final_score >= 0.7:       # High value -> Promote directly to Long-Term Memory
                promoted_long_term.append(m)
            elif final_score >= 0.35:    # Medium value -> Keep in active Working Memory
                kept_working.append(m)
            else:                        # Low value -> Evicted automatically to drop token size
                pass

        # Sync routed allocations back to the main memory layer arrays
        memory_system.long_term_memory = promoted_long_term
        memory_system.working_memory = kept_working