# better-scraper

A Python library to search the web using DuckDuckGo, rerank results with LLMs (OpenAI/Gemini), and optionally generate summarized answers from web content.

## Features
- Web search via DuckDuckGo (DDGS)
- Query expansion using LLMs (OpenAI, Gemini)
- Reranking with keyword overlap or LLM embeddings
- Deduplication of results
- Two search modes: `basic` (fast, no reranking) and `advanced` (reranking)
- Optional answer generation from top web results using LLMs
- Proxy and timeout support for DDGS

## Installation

```bash
pip install better-scraper
```

## Usage

### Basic Web Search
```python
from better_scraper.web_search import search_web

results = search_web("What is quantum computing?", mode="basic")
for r in results["results"]:
    print(r["title"], r["url"])
```

### Advanced Search with OpenAI Reranking
```python
from better_scraper.web_search import search_web
from better_scraper.llm.openai import OpenAIEngine

engine = OpenAIEngine(api_key="YOUR_OPENAI_API_KEY")
output = search_web(
    "What is quantum computing?",
    llm_engine=engine,
    mode="advanced"
)
for r in output["results"]:
    print(r["title"], r["url"], r["relevancy_score"])
```

### Generate an Answer from Web Results
```python
output = search_web(
    "What is quantum computing?",
    llm_engine=engine,
    mode="advanced",
    provide_answer=True
)
print("Answer:", output["answer"])
```

### Use a Proxy
```python
output = search_web(
    "What is quantum computing?",
    proxy="socks5h://user:password@geo.iproyal.com:32325",
    timeout=20
)
```

### Use Gemini LLM Engine
```python
from better_scraper.llm.gemini import GeminiEngine
engine = GeminiEngine(api_key="YOUR_GEMINI_API_KEY")
output = search_web(
    "What is quantum computing?",
    llm_engine=engine,
    mode="advanced",
    provide_answer=True
)
print(output["answer"])
```

## FastAPI Server & Docker Deployment

You can deploy a FastAPI server for this search engine using Docker Compose.

### 1. Build and Run with Docker Compose
```bash
cd docker
# Set your API keys in the environment if needed
export OPENAI_API_KEY=sk-...  # optional, for OpenAI
export GEMINI_API_KEY=...     # optional, for Gemini

docker-compose up --build
```
The API will be available at [http://localhost:8000](http://localhost:8000)

### 2. Sample cURL Request
```bash
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is quantum computing?",
    "llm_engine": "openai",
    "provide_answer": true
  }'
```

### 3. Sample Response
```json
{
  "results": [
    {"title": "Quantum computing - Wikipedia", "snippet": "Quantum computing is...", "url": "https://en.wikipedia.org/...", ...},
    ...
  ],
  "answer": "Quantum computing is ... [summary from LLM]"
}
```

## Parameters
- `query` (str): The user's search query.
- `max_results` (int): Max results per query (default: 10).
- `llm_engine` (LLMEngine): Optional. Use OpenAIEngine or GeminiEngine for query expansion, reranking, and answer generation.
- `num_queries` (int): Number of expanded queries to generate (default: 3).
- `mode` (str): 'basic' (no reranking) or 'advanced' (reranking, default: 'advanced').
- `proxy` (str): Optional. Proxy for DDGS (e.g., socks5h://user:pass@host:port).
- `timeout` (int): DDGS timeout in seconds (default: 20).
- `provide_answer` (bool): If True, generate an answer from the top 5 results using the LLM engine.

## Output
- Always returns a dict with a `results` key (list of web results).
- If `provide_answer=True`, also includes an `answer` key (string).

## Testing
Run tests with:
```bash
pytest tests/
```

## License
MIT 