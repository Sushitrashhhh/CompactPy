import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sentence_transformers import SentenceTransformer

class MemoryClusteringEngine:
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        """
        Initializes the Phase 2 Machine Learning Clustering Engine using
        SentenceTransformers and scikit-learn clustering algorithms.
        """
        self.model = SentenceTransformer(embedding_model_name)

    def _get_embeddings(self, texts: list[str]) -> np.ndarray:
        """Converts raw context text arrays into normalized dense vectors."""
        if not texts:
            return np.empty((0, 0))
        return np.array(self.model.encode(texts))
    
    def cluster_with_kmeans(self, texts: list[str], n_clusters: int = 5) -> dict[int, list[str]]:
        '''
        partitions memory embeddings into k distinct topical centroids.
        excellent for forcing a fixed number of clusters for structured memory retrieval.
        '''
        if len(texts)< n_clusters:
            return{0: texts}  # Return all texts in a single cluster if fewer than n_clusters
        
        embeddings = self._get_embeddings(texts)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
        labels = kmeans.fit_predict(embeddings)

        clustered_memory = {i: [] for i in range(n_clusters)}
        for idx, label in enumerate(labels):
            clustered_memory[label].append(texts[idx]) 
        return clustered_memory
    
    def cluster_by_dbscan(self, texts: list[str], eps: float = 0.4, min_samples: int = 2) -> dict[int, list[str]]:
        '''
        groups memory embeddings based on density and proximity.
        ideal for discovering natural clusters without predefining cluster counts.
        '''
        if not texts:
            return {}
        
        embeddings = self._get_embeddings(texts)
        dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
        labels = dbscan.fit_predict(embeddings)

        clustered_memory = {"noise":[]}
        for idx, label in enumerate(labels):
            if label == -1:
                continue  # Skip noise points
            if label not in clustered_memory:
                clustered_memory[label] = []
            clustered_memory[label].append(texts[idx])
        
        return clustered_memory