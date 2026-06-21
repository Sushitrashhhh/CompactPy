import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sentence_transformers import SentenceTransformer

class MemoryRankingEngine:
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        """
        Initializes the Phase 2 ML-driven feature engineering and ranking system.
        Replaces hardcoded scoring heuristics with a trained regressor.
        """
        self.similarity_model = SentenceTransformer(embedding_model_name)
        # Random Forest handles non-linear interactions between statistical weights cleanly
        self.model = RandomForestRegressor(n_estimators=50, random_state=42)
        self.is_trained = False

    def _calculate_semantic_relevance(self, query: str, texts: list[str]) -> np.ndarray:
        """Computes direct cosine similarity between an active query and context blocks."""
        if not texts:
            return np.array([])
        query_emb = self.similarity_model.encode(query)
        text_embs = self.similarity_model.encode(texts)
        
        # Normalize vectors for algebraic cosine similarity calculations
        norm_query = query_emb / np.linalg.norm(query_emb)
        norm_texts = text_embs / np.linalg.norm(text_embs, axis=1, keepdims=True)
        return np.dot(norm_texts, norm_query)

    def build_feature_matrix(self, query: str, memories: list[dict]) -> np.ndarray:
        """
        Engineers a composite feature array for the machine learning pipeline.
        Features compiled: [Importance, Utility, Recency, Frequency, Semantic Relevance]
        """
        if not memories:
            return np.empty((0, 5))
            
        texts = [m["text"] for m in memories]
        semantic_scores = self._calculate_semantic_relevance(query, texts)
        
        features = []
        for idx, m in enumerate(memories):
            features.append([
                m.get("importance", 0.5),
                m.get("utility", 0.5),
                m.get("recency", 1.0),
                m.get("frequency", 1.0),
                semantic_scores[idx]
            ])
        return np.array(features)

    def train_baseline_ranker(self):
        """Trains the internal scoring network on a more robust profile matrix."""
        # Features: [Importance, Utility, Recency, Frequency, Semantic Relevance]
        X_train = np.array([
            [0.9, 0.8, 0.9, 5, 0.85],  # Explicit critical technical match -> Keep (1.0)
            [0.8, 0.7, 0.4, 9, 0.92],  # Historically deep context pillar -> Keep (0.95)
            [0.2, 0.1, 0.2, 1, 0.10],  # Intermittent noisy data line -> Evict (0.0)
            [0.4, 0.3, 0.9, 1, 0.15],  # Fresh memory but ZERO query relevance -> Evict (0.0)
            [0.3, 0.2, 0.95, 2, 0.05], # Very recent off-topic line -> Evict (0.0)
            [0.5, 0.4, 0.7, 2, 0.45],  # Standard baseline comment item -> Mid-Tier (0.45)
        ])
        y_train = np.array([1.0, 0.95, 0.0, 0.0, 0.0, 0.45])
        
        self.model.fit(X_train, y_train)
        self.is_trained = True

    def predict_importance_scores(self, query: str, memories: list[dict]) -> np.ndarray:
        """Evaluates ongoing context segments to output statistical importance ranking scores."""
        if not memories:
            return np.array([])
            
        if not self.is_trained:
            self.train_baseline_ranker()
            
        X = self.build_feature_matrix(query, memories)
        return self.model.predict(X)