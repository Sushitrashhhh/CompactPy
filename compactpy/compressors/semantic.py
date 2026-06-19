from sentence_transformers import SentenceTransformer
import numpy as np
from compactpy.token_utils import calculate_savings

class SemanticCompressor:
    '''
    v2: Version 2: Semantic Compression Engine.
    Filters out semantically redundant text items based on an embedding similarity threshold.'''

    def __init__(self, threshold: float = 0.75, model_name: str='all-MiniLM-L6-v2'):
        self.threshold = threshold
        self.model = SentenceTransformer(model_name)

    @staticmethod
    def _cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        dot_product = np.dot(v1, v2)
        norm_v1 = np.linalg.norm(v1)
        norm_v2 = np.linalg.norm(v2)
        if norm_v1 == 0 or norm_v2 == 0:
            return 0.0
        return float(dot_product / (norm_v1 * norm_v2))
    
    def compress(self, texts: list[str]) -> tuple[list[str], dict]:
        if not texts:
            return [], {"original_length": 0, "compressed_length": 0, "savings_percentage": 0.0}
        
        #compute embeddings for all texts
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        unique_texts = []
        unique_embeddings = []

        for idx, text in enumerate(texts):
            current_emb = embeddings[idx]
            is_redundant = False

            for unique_emb in unique_embeddings:
                similarity = self._cosine_similarity(current_emb, unique_emb)
                if similarity >= self.threshold:
                    is_redundant = True
                    break

            if not is_redundant:
                unique_texts.append(text)
                unique_embeddings.append(current_emb)
        
        # Generate a report on token savings
        original_text = " ".join(texts)
        compressed_text = " ".join(unique_texts)
        report = calculate_savings(original_text, compressed_text)
        return unique_texts, report