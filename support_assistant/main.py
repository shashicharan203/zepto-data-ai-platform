from fastapi import FastAPI
from pydantic import BaseModel

from graph import ask_question


app = FastAPI(
    title="Zepto Support Assistant",
    description="RAG-based Zepto policy support service",
    version="1.0.0"
)


class AskRequest(BaseModel):
    query: str


class AskResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float


@app.get("/")
def home():
    return {
        "message": "Zepto Support Assistant is running"
    }


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    return ask_question(request.query)