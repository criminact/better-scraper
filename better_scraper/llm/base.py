from abc import ABC, abstractmethod
from typing import List, Dict

class LLMEngine(ABC):
    @abstractmethod
    def generate_queries(self, user_query: str, num_queries: int = 3) -> List[str]:
        """
        Generate multiple related queries from the user's original query.
        """
        pass

    @abstractmethod
    def get_embedding(self, text: str) -> List[float]:
        """
        Get an embedding vector for the given text.
        """
        pass

    @abstractmethod
    def generate_answer(self, prompt: str) -> str:
        """
        Generate an answer to the user's query based on the provided prompt (web content).
        """
        pass

    @abstractmethod
    def answer_from_web_results(self, query: str, results: List[Dict], top_n: int = 5) -> str:
        """
        Build a prompt from the top N web results and the user query, then call generate_answer.
        """
        pass 