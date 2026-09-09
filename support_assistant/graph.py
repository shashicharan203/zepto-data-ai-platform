import os
import json
from typing import TypedDict

import requests
import chromadb
from pydantic import BaseModel, Field, ValidationError
from sentence_transformers import SentenceTransformer
from langgraph.graph import StateGraph, START, END


MOCK_LLM = os.getenv("MOCK_LLM", "1") != "0"

DB_DIR = "chroma_db"
COLLECTION_NAME = "zepto_policies"

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path=DB_DIR)

collection = client.get_collection(
    name=COLLECTION_NAME
)


class AnswerResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float = Field(ge=0, le=1)


class GraphState(TypedDict, total=False):
    query: str
    intent: str
    context: list[str]
    source_ids: list[str]
    response: dict


PROMPT_TEMPLATE = """
Role:
You are a Zepto customer support assistant.

Context:
Use only the retrieved Zepto policy context provided below.

Task:
Answer the customer's question accurately using only the provided context.

Format:
Return valid JSON with exactly these fields:
answer: string
sources: list of document IDs
confidence: number between 0 and 1

Length:
Keep the answer concise and directly relevant.

Negative constraint:
Do not answer using information that is not present in the provided context.
Do not invent Zepto policies, prices, timings, or rules.

Few-shot example:
Question: What is the delivery fee for orders below INR 149?
Context: Orders below INR 149 incur a flat INR 25 delivery fee.
Answer:
{"answer":"Orders below INR 149 incur a INR 25 delivery fee.","sources":["doc_01"],"confidence":1.0}

Customer Question:
{query}

Retrieved Context:
{context}
"""


def call_real_llm(prompt):
    api_key = os.getenv("GROQ_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is required when MOCK_LLM=0."
        )

    model = os.getenv(
        "GROQ_MODEL",
        "llama-3.3-70b-versatile"
    )

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        json={
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "temperature": 0
        },
        timeout=30
    )

    response.raise_for_status()

    data = response.json()

    return data["choices"][0]["message"]["content"]


def classify_intent(state: GraphState):
    query = state["query"]
    query_lower = query.lower()

    keywords = [
        "delivery",
        "return",
        "refund",
        "membership",
        "tracking",
        "cancel",
        "gift card",
        "support hours"
    ]

    if MOCK_LLM:
        if any(keyword in query_lower for keyword in keywords):
            intent = "policy_question"
        else:
            intent = "general_question"

    else:
        prompt = f"""
Classify the following customer query.

Return exactly one value:
policy_question
or
general_question

Use policy_question if the query is about Zepto delivery,
returns, refunds, membership, tracking, cancellation,
gift cards, or support hours.

Query:
{query}
"""

        raw = call_real_llm(prompt).strip().lower()

        if "policy_question" in raw:
            intent = "policy_question"
        else:
            intent = "general_question"

    return {
        "intent": intent
    }


def retrieve_and_answer(state: GraphState):
    query = state["query"]

    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=3
    )

    documents = results["documents"][0]
    source_ids = results["ids"][0]

    if MOCK_LLM:

        top_chunk_snippet = documents[0][:200]

        answer = (
            f"Based on the retrieved context: "
            f"{top_chunk_snippet}"
        )

        response = AnswerResponse(
            answer=answer,
            sources=source_ids,
            confidence=1.0
        )

        return {
            "context": documents,
            "source_ids": source_ids,
            "response": response.model_dump()
        }

    context = "\n\n".join(
        f"[{source_id}] {document}"
        for source_id, document in zip(
            source_ids,
            documents
        )
    )

    prompt = PROMPT_TEMPLATE.format(
        query=query,
        context=context
    )

    last_error = None

    for attempt in range(3):

        try:

            current_prompt = prompt

            if attempt > 0:
                current_prompt += """

Your previous response failed validation.

Return ONLY valid JSON matching this schema:

{
  "answer": "string",
  "sources": ["document_id"],
  "confidence": 0.0
}

Do not include markdown or additional text.
"""

            raw_output = call_real_llm(
                current_prompt
            )

            parsed = json.loads(raw_output)

            response = AnswerResponse.model_validate(
                parsed
            )

            return {
                "context": documents,
                "source_ids": source_ids,
                "response": response.model_dump()
            }

        except (
            json.JSONDecodeError,
            ValidationError,
            KeyError,
            TypeError
        ) as error:

            last_error = error

    error_response = AnswerResponse(
        answer=(
            "ERROR: The real LLM response failed "
            "structured-output validation after 3 attempts."
        ),
        sources=[],
        confidence=0.0
    )

    return {
        "context": documents,
        "source_ids": source_ids,
        "response": error_response.model_dump()
    }


def direct_answer(state: GraphState):

    query = state["query"]

    if MOCK_LLM:

        response = AnswerResponse(
            answer=(
                "I can only answer questions about "
                "Zepto policies right now."
            ),
            sources=[],
            confidence=1.0
        )

        return {
            "response": response.model_dump()
        }

    prompt = f"""
Role:
You are a Zepto customer support assistant.

Task:
Answer the following general customer question.

Format:
Return plain text only.

Length:
Keep the answer concise.

Question:
{query}
"""

    last_error = None

    for attempt in range(3):

        try:

            answer = call_real_llm(prompt).strip()

            response = AnswerResponse(
                answer=answer,
                sources=[],
                confidence=0.8
            )

            return {
                "response": response.model_dump()
            }

        except Exception as error:

            last_error = error

    error_response = AnswerResponse(
        answer=(
            "ERROR: The real LLM response could not "
            "be generated after 3 attempts."
        ),
        sources=[],
        confidence=0.0
    )

    return {
        "response": error_response.model_dump()
    }


def route_intent(state: GraphState):

    if state["intent"] == "policy_question":
        return "retrieve_and_answer"

    return "direct_answer"


builder = StateGraph(GraphState)

builder.add_node(
    "classify_intent",
    classify_intent
)

builder.add_node(
    "retrieve_and_answer",
    retrieve_and_answer
)

builder.add_node(
    "direct_answer",
    direct_answer
)

builder.add_edge(
    START,
    "classify_intent"
)

builder.add_conditional_edges(
    "classify_intent",
    route_intent,
    {
        "retrieve_and_answer": "retrieve_and_answer",
        "direct_answer": "direct_answer"
    }
)

builder.add_edge(
    "retrieve_and_answer",
    END
)

builder.add_edge(
    "direct_answer",
    END
)

graph = builder.compile()


def ask_question(query: str) -> AnswerResponse:

    result = graph.invoke({
        "query": query
    })

    return AnswerResponse.model_validate(
        result["response"]
    )


if __name__ == "__main__":

    policy_query = (
        "What is the delivery fee for orders below INR 149?"
    )

    general_query = (
        "What is the capital of India?"
    )

    print("\nPolicy Question:")
    print(
        ask_question(
            policy_query
        ).model_dump_json(indent=2)
    )

    print("\nGeneral Question:")
    print(
        ask_question(
            general_query
        ).model_dump_json(indent=2)
    )