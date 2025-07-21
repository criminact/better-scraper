from better_scraper.search import DDGSSearcher
from typing import Dict, Any

def search_web(query: str, max_results: int = 10, llm_engine=None, num_queries: int = 3, mode: str = 'advanced', proxy: str = None, timeout: int = 20, provide_answer: bool = False) -> Dict[str, Any]:
    """
    Search the web using DuckDuckGo and return a list of results.
    Uses the provided llm_engine to generate multiple queries from the user's query.
    Each result is a dict with 'title', 'snippet', and 'url'.
    Results are deduplicated and reranked by relevancy to the original query in 'advanced' mode.
    In 'basic' mode, results are only deduplicated.
    If llm_engine is OpenAIEngine or GeminiEngine, use embedding-based reranking. Otherwise, use keyword overlap.
    Optionally, provide a proxy for DDGS (e.g., socks5h://user:password@host:port).
    If provide_answer is True, use the LLM engine to generate an answer from the top 5 results after reranking.
    Returns a dict with 'results' and, if requested, 'answer'.
    """
    searcher = DDGSSearcher(proxy=proxy, timeout=timeout)
    results = searcher.search(query, max_results=max_results, llm_engine=llm_engine, num_queries=num_queries, mode=mode)

    output = {'results': results}

    if provide_answer and llm_engine is not None and results:
        answer = llm_engine.answer_from_web_results(query, results, top_n=5)
        output['answer'] = answer

    return output 