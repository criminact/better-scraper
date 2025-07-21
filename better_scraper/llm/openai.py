import os
from typing import List, Dict
from .base import LLMEngine

try:
    import openai
except ImportError:
    openai = None

class OpenAIEngine(LLMEngine):
    def __init__(self, api_key: str = None, model: str = "gpt-3.5-turbo", embedding_model: str = "text-embedding-ada-002"):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        self.embedding_model = embedding_model

    def generate_queries(self, user_query: str, num_queries: int = 3) -> List[str]:
        prompt = (
            f"Given the user query: '{user_query}', generate {num_queries} different but related search queries "
            "that would help gather comprehensive information on the topic. Return them as a list."
        )
        # TODO: Replace the following with actual OpenAI API call
        # Example placeholder response:
        return [f"{user_query} (variation {i+1})" for i in range(num_queries)]

    def get_embedding(self, text: str) -> List[float]:
        if openai is None:
            raise ImportError("openai package is required for OpenAIEngine embedding.")
        openai.api_key = self.api_key
        response = openai.Embedding.create(input=[text], model=self.embedding_model)
        return response['data'][0]['embedding']

    def generate_answer(self, prompt: str) -> str:
        if openai is None:
            raise ImportError("openai package is required for OpenAIEngine answer generation.")
        openai.api_key = self.api_key
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response['choices'][0]['message']['content'].strip()

    def answer_from_web_results(self, query: str, results: List[Dict], top_n: int = 5) -> str:
        top_results = results[:top_n]
        context = "\n\n".join([
            f"Title: {r.get('title', '')}\nSnippet: {r.get('snippet', '')}" for r in top_results
        ])
        prompt = (
            f"Based on the following web search results, answer the user's query: '{query}'.\n"
            f"Web Results:\n{context}\n"
            "Provide a concise, accurate, and well-cited answer."
        )
        return self.generate_answer(prompt) 