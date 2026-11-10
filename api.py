from typing import Literal, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from rag import LLMError, answer

app = FastAPI(title="RAG Assistant")


class Question(BaseModel):
    question: str = Field(min_length=3)
    k: int = Field(4, ge=1, le=10)
    mode: Literal["dense", "bm25", "hybrid"] = "dense"
    rerank: bool = True
    provider: Optional[Literal["ollama", "claude"]] = None  # None = variable LLM_PROVIDER


class Source(BaseModel):
    source: str
    page: int


class Answer(BaseModel):
    answer: str
    sources: list[Source]


@app.post("/ask", response_model=Answer)
def ask(q: Question):
    try:
        return answer(q.question, q.k, q.mode, q.rerank, q.provider)
    except LLMError as e:
        raise HTTPException(503, str(e))
