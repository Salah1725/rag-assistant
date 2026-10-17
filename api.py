from fastapi import FastAPI
from pydantic import BaseModel
from rag import answer

app = FastAPI(title="RAG Assistant")

class Question(BaseModel):
    question: str
    k: int = 4

class Source(BaseModel):
    source: str
    page: int

class Answer(BaseModel):
    answer: str
    sources: list[Source]

@app.post("/ask", response_model=Answer)
def ask(q: Question):
    return answer(q.question, q.k)