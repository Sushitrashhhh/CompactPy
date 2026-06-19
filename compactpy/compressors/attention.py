from sentence_transformers import SentenceTransformer
import numpy as np
from compactpy.token_utils import calculate_savings

class AttentionAwareCompressor:
    """
    Version 6: Attention-Aware Dynamic Compression Framework.
    Predicts and isolates key context dependencies relative to an incoming query token budget.
    """
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    @staticmethod
    def _cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        return float(dot_product / (norm_v1 * norm_v2))

    def compress_context_for_query(self, query: str, context_pool: list[str], token_budget: int = 40) -> tuple[str, dict]:
        """
        Predicts importance weights relative to a live query and filters text down to fit a target token budget.
        """
        if not context_pool:
            return "", {}

        # 1. Embed the query and the historical context records
        query_embedding = self.model.encode(query, convert_to_numpy=True)
        context_embeddings = self.model.encode(context_pool, convert_to_numpy=True)
        
        ranked_memories = []

        # 2. Score relevance using vector closeness metrics
        for idx, text in enumerate(context_pool):
            similarity = self._cosine_similarity(query_embedding, context_embeddings[idx])
            ranked_memories.append((similarity, text))

        # Sort context blocks dynamically (highest attention relevance first)
        ranked_memories.sort(key=lambda x: x[0], reverse=True)

        selected_blocks = []
        current_tokens = 0

        # 3. Fill the prompt token budget safely using tiktoken metrics checks
        from compactpy.token_utils import count_tokens
        
        for similarity, text in ranked_memories:
            block_tokens = count_tokens(text)
            # Retain high-attention vectors if they fit inside our target token limit
            if current_tokens + block_tokens <= token_budget:
                selected_blocks.append(text)
                current_tokens += block_tokens
            else:
                continue

        original_flat = " ".join(context_pool)
        compressed_flat = " ".join(selected_blocks)
        metrics = calculate_savings(original_flat, compressed_flat)

        return compressed_flat, metrics