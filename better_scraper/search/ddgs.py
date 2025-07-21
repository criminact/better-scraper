from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional
from ddgs import DDGS
from .base import BaseSearcher
from .rerank import KeywordReranker, EmbeddingReranker
from better_scraper.llm.openai import OpenAIEngine
from better_scraper.llm.gemini import GeminiEngine

class DDGSSearcher(BaseSearcher):
    def __init__(self, proxy: Optional[str] = None, timeout: int = 20):
        self.proxy = proxy
        self.timeout = timeout

    def _ddgs_search_single(self, query: str, max_results: int) -> List[Dict]:
        results = []
        with DDGS(proxy=self.proxy, timeout=self.timeout) as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    'title': r.get('title'),
                    'snippet': r.get('body'),
                    'url': r.get('href')
                })
        return results

    def _deduplicate_results(self, results: List[Dict]) -> List[Dict]:
        seen_urls = set()
        deduped = []
        for r in results:
            url = r.get('url')
            if url and url not in seen_urls:
                deduped.append(r)
                seen_urls.add(url)
        return deduped

    def search(self, query: str, max_results: int = 10, llm_engine=None, num_queries: int = 3, mode: str = 'advanced') -> List[Dict]:
        if llm_engine is not None:
            queries = llm_engine.generate_queries(query, num_queries=num_queries)
        else:
            queries = [query]

        all_results = []
        with ThreadPoolExecutor() as executor:
            future_to_query = {
                executor.submit(self._ddgs_search_single, q, max_results): q for q in queries
            }
            for future in as_completed(future_to_query):
                all_results.extend(future.result())

        deduped = self._deduplicate_results(all_results)

        if mode == 'basic':
            return deduped

        if isinstance(llm_engine, (OpenAIEngine, GeminiEngine)):
            reranker = EmbeddingReranker(llm_engine)
        else:
            reranker = KeywordReranker()
        reranked = reranker.rerank(deduped, query)
        return reranked 