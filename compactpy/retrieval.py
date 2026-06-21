from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np
from sentence_transformers import SentenceTransformer


class BM25RetrievalEngine:
    def __init__(self, k1: float = 1.2, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.corpus: list[str] = []
        self.term_frequencies: list[Counter[str]] = []
        self.document_frequencies: Counter[str] = Counter()
        self.document_lengths: list[int] = []
        self.avg_document_length = 0.0

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"\b\w+\b", text.lower())

    def fit(self, corpus: list[str]) -> None:
        self.corpus = list(corpus)
        self.term_frequencies = []
        self.document_frequencies = Counter()
        self.document_lengths = []

        for document in self.corpus:
            tokens = self._tokenize(document)
            frequencies = Counter(tokens)
            self.term_frequencies.append(frequencies)
            self.document_lengths.append(len(tokens))
            self.document_frequencies.update(frequencies.keys())

        if self.document_lengths:
            self.avg_document_length = sum(self.document_lengths) / len(self.document_lengths)
        else:
            self.avg_document_length = 0.0

    def _idf(self, token: str) -> float:
        total_documents = len(self.corpus)
        document_frequency = self.document_frequencies.get(token, 0)
        return math.log(1 + (total_documents - document_frequency + 0.5) / (document_frequency + 0.5))

    def _score_document(self, query_tokens: list[str], document_index: int) -> float:
        document_frequencies = self.term_frequencies[document_index]
        document_length = self.document_lengths[document_index]
        score = 0.0

        for token in query_tokens:
            term_frequency = document_frequencies.get(token, 0)
            if term_frequency == 0:
                continue

            idf = self._idf(token)
            numerator = term_frequency * (self.k1 + 1)
            denominator = term_frequency + self.k1 * (1 - self.b + self.b * document_length / max(self.avg_document_length, 1.0))
            score += idf * (numerator / denominator)

        return score

    def search(self, query: str, corpus: list[str] | None = None, top_n: int = 5) -> list[tuple[str, float]]:
        if corpus is not None:
            self.fit(corpus)
        elif not self.corpus:
            return []

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return [(document, 0.0) for document in self.corpus[:top_n]]

        ranked_documents = [
            (self.corpus[index], self._score_document(query_tokens, index))
            for index in range(len(self.corpus))
        ]
        ranked_documents.sort(key=lambda item: item[1], reverse=True)
        return ranked_documents[:top_n]


class HybridRetrievalEngine:
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2", k1: float = 1.2, b: float = 0.75):
        self.sparse_engine = BM25RetrievalEngine(k1=k1, b=b)
        self.dense_model = SentenceTransformer(embedding_model_name)

    def _calculate_dense_scores(self, query: str, corpus: list[str]) -> np.ndarray:
        query_embedding = self.dense_model.encode(query, convert_to_numpy=True)
        corpus_embeddings = self.dense_model.encode(corpus, convert_to_numpy=True)

        query_norm = np.linalg.norm(query_embedding)
        corpus_norms = np.linalg.norm(corpus_embeddings, axis=1)

        if query_norm == 0:
            return np.zeros(len(corpus), dtype=float)

        safe_corpus_norms = np.where(corpus_norms == 0, 1.0, corpus_norms)
        normalized_query = query_embedding / query_norm
        normalized_corpus = corpus_embeddings / safe_corpus_norms[:, None]
        return np.dot(normalized_corpus, normalized_query)

    def search_hybrid(self, query: str, corpus: list[str], top_n: int = 2) -> list[tuple[str, float]]:
        if not corpus:
            return []

        self.sparse_engine.fit(corpus)
        sparse_results = self.sparse_engine.search(query, top_n=len(corpus))
        sparse_ranked = [text for text, _ in sparse_results]

        dense_scores = self._calculate_dense_scores(query, corpus)
        dense_ranked_indices = np.argsort(dense_scores)[::-1]
        dense_ranked = [corpus[index] for index in dense_ranked_indices]

        rrf_scores: dict[str, float] = {}
        for document in corpus:
            sparse_rank = sparse_ranked.index(document) + 1 if document in sparse_ranked else len(corpus) + 1
            dense_rank = dense_ranked.index(document) + 1
            rrf_scores[document] = (1.0 / (60 + sparse_rank)) + (1.0 / (60 + dense_rank))

        ranked_documents = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
        return ranked_documents[:top_n]