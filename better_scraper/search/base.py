from abc import ABC, abstractmethod
from typing import List, Dict

class BaseSearcher(ABC):
    @abstractmethod
    def search(self, query: str, max_results: int = 10, llm_engine=None, num_queries: int = 3) -> List[Dict]:
        """
        Search the web and return a list of results.
        """
        pass 