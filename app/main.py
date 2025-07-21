from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import os
from better_scraper.web_search import search_web
from better_scraper.llm.openai import OpenAIEngine
from better_scraper.llm.gemini import GeminiEngine

app = FastAPI(title="Better Scraper Search API")

class SearchRequest(BaseModel):
    query: str
    max_results: Optional[int] = 10
    llm_engine: Optional[str] = None  # 'openai', 'gemini', or None
    num_queries: Optional[int] = 3
    mode: Optional[str] = 'advanced'  # 'basic' or 'advanced'
    proxy: Optional[str] = None
    timeout: Optional[int] = 20
    provide_answer: Optional[bool] = False
    openai_api_key: Optional[str] = None
    gemini_api_key: Optional[str] = None

@app.post("/search")
def search(request: SearchRequest) -> Dict[str, Any]:
    llm_engine = None
    if request.llm_engine == 'openai':
        api_key = request.openai_api_key or os.getenv("OPENAI_API_KEY")
        llm_engine = OpenAIEngine(api_key=api_key)
    elif request.llm_engine == 'gemini':
        api_key = request.gemini_api_key or os.getenv("GEMINI_API_KEY")
        llm_engine = GeminiEngine(api_key=api_key)
    
    return search_web(
        query=request.query,
        max_results=request.max_results,
        llm_engine=llm_engine,
        num_queries=request.num_queries,
        mode=request.mode,
        proxy=request.proxy,
        timeout=request.timeout,
        provide_answer=request.provide_answer
    )

@app.get("/")
def root():
    return {"message": "Better Scraper Search API. Use POST /search with your query."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=False) 