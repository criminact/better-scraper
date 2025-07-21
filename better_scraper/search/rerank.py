from abc import ABC, abstractmethod
from typing import List, Dict
import re
import numpy as np

class Reranker(ABC):
    @abstractmethod
    def rerank(self, results: List[Dict], user_query: str) -> List[Dict]:
        pass

class KeywordReranker(Reranker):
    def rerank(self, results: List[Dict], user_query: str) -> List[Dict]:
        def _keyword_overlap_score(query: str, text: str) -> int:
            query_words = set(re.findall(r'\w+', query.lower()))
            text_words = set(re.findall(r'\w+', text.lower()))
            return len(query_words & text_words)
        for result in results:
            combined_text = (result.get('title') or '') + ' ' + (result.get('snippet') or '')
            result['relevancy_score'] = _keyword_overlap_score(user_query, combined_text)
        return sorted(results, key=lambda r: r['relevancy_score'], reverse=True)

class EmbeddingReranker(Reranker):
    def __init__(self, llm_engine):
        self.llm_engine = llm_engine
    def rerank(self, results: List[Dict], user_query: str) -> List[Dict]:
        query_emb = np.array(self.llm_engine.get_embedding(user_query))
        for result in results:
            combined_text = (result.get('title') or '') + ' ' + (result.get('snippet') or '')
            result_emb = np.array(self.llm_engine.get_embedding(combined_text))
            sim = float(np.dot(query_emb, result_emb) / (np.linalg.norm(query_emb) * np.linalg.norm(result_emb) + 1e-8))
            result['relevancy_score'] = sim
        return sorted(results, key=lambda r: r['relevancy_score'], reverse=True) 