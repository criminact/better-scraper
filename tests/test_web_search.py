import os
import sys
import pytest

# Ensure local package is used for testing
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from better_scraper.web_search import search_web
from better_scraper.llm.openai import OpenAIEngine
from better_scraper.llm.gemini import GeminiEngine

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

@pytest.mark.parametrize("mode", ["basic", "advanced"])
def test_search_web_results(mode):
    results = search_web("What is quantum computing?", mode=mode)["results"]
    assert isinstance(results, list)
    assert len(results) > 0
    assert all("title" in r and "url" in r for r in results)

@pytest.mark.skipif(not OPENAI_API_KEY, reason="No OpenAI API key set")
def test_search_web_with_openai_advanced():
    engine = OpenAIEngine(api_key=OPENAI_API_KEY)
    output = search_web("What is quantum computing?", llm_engine=engine, mode="advanced")
    results = output["results"]
    assert isinstance(results, list)
    assert len(results) > 0
    assert all("title" in r and "url" in r for r in results)

@pytest.mark.skipif(not OPENAI_API_KEY, reason="No OpenAI API key set")
def test_search_web_with_answer():
    engine = OpenAIEngine(api_key=OPENAI_API_KEY)
    output = search_web("What is quantum computing?", llm_engine=engine, mode="advanced", provide_answer=True)
    assert "results" in output
    assert "answer" in output
    assert isinstance(output["answer"], str)
    assert len(output["answer"]) > 0

@pytest.mark.skipif(not GEMINI_API_KEY, reason="No Gemini API key set")
def test_search_web_with_gemini_advanced():
    try:
        import google.generativeai
    except ImportError:
        pytest.skip("google-generativeai not installed")
    engine = GeminiEngine(api_key=GEMINI_API_KEY)
    output = search_web("What is quantum computing?", llm_engine=engine, mode="advanced")
    results = output["results"]
    assert isinstance(results, list)
    assert len(results) > 0
    assert all("title" in r and "url" in r for r in results)

@pytest.mark.skipif(not GEMINI_API_KEY, reason="No Gemini API key set")
def test_search_web_with_gemini_answer():
    try:
        import google.generativeai
    except ImportError:
        pytest.skip("google-generativeai not installed")
    engine = GeminiEngine(api_key=GEMINI_API_KEY)
    output = search_web("What is quantum computing?", llm_engine=engine, mode="advanced", provide_answer=True)
    assert "results" in output
    assert "answer" in output
    assert isinstance(output["answer"], str)
    assert len(output["answer"]) > 0 