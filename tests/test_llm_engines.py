import os
import sys
import pytest

# Ensure local package is used for testing
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from better_scraper.llm.openai import OpenAIEngine
from better_scraper.llm.gemini import GeminiEngine

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

@pytest.mark.skipif(not OPENAI_API_KEY, reason="No OpenAI API key set")
def test_openai_engine_chat_and_embedding():
    engine = OpenAIEngine(api_key=OPENAI_API_KEY)
    # Test chat (generate_answer)
    prompt = "What is the capital of France?"
    answer = engine.generate_answer(prompt)
    assert isinstance(answer, str)
    assert "Paris".lower() in answer.lower()
    # Test embedding
    emb = engine.get_embedding("test sentence for embedding")
    assert isinstance(emb, list)
    assert len(emb) > 0

@pytest.mark.skipif(not GEMINI_API_KEY, reason="No Gemini API key set")
def test_gemini_engine_chat_and_embedding():
    try:
        import google.generativeai
    except ImportError:
        pytest.skip("google-generativeai not installed")
    engine = GeminiEngine(api_key=GEMINI_API_KEY)
    # Test chat (generate_answer)
    prompt = "What is the capital of France?"
    answer = engine.generate_answer(prompt)
    assert isinstance(answer, str)
    assert "Paris".lower() in answer.lower()
    # Test embedding
    emb = engine.get_embedding("test sentence for embedding")
    assert isinstance(emb, list)
    assert len(emb) > 0 