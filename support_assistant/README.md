
# Zepto Support Assistant

## Overview

This module implements a Retrieval-Augmented Generation (RAG) support assistant for Zepto policy questions.

The application:

* Loads 8 Zepto policy documents
* Generates document embeddings using Sentence Transformers
* Uses `all-MiniLM-L6-v2` as the embedding model
* Stores embeddings in ChromaDB
* Uses cosine similarity for retrieval
* Uses LangGraph for intent routing and workflow orchestration
* Retrieves the top 3 relevant policy documents for policy questions
* Supports an offline deterministic `MOCK_LLM` mode
* Supports an optional real LLM through the configured Groq API
* Uses Pydantic for structured response validation
* Exposes the assistant through a FastAPI `/ask` endpoint
* Includes Docker configuration for containerization

The graded baseline runs completely offline for LLM generation when `MOCK_LLM` is enabled.

---

# Project Structure

```text
support_assistant/
│
├── docs/
│   ├── doc_01.txt
│   ├── doc_02.txt
│   ├── doc_03.txt
│   ├── doc_04.txt
│   ├── doc_05.txt
│   ├── doc_06.txt
│   ├── doc_07.txt
│   └── doc_08.txt
│
├── chroma_db/
│
├── ingest.py
├── graph.py
├── main.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
└── README.md
```

---

# 1. Technology Stack

| Component             | Technology            |
| --------------------- | --------------------- |
| Language              | Python                |
| Embeddings            | Sentence Transformers |
| Embedding Model       | `all-MiniLM-L6-v2`    |
| Vector Database       | ChromaDB              |
| Similarity            | Cosine similarity     |
| Orchestration         | LangGraph             |
| Structured Validation | Pydantic              |
| API                   | FastAPI               |
| Server                | Uvicorn               |
| Optional LLM          | Groq API              |
| Containerization      | Docker                |

---

# 2. Policy Documents

The assistant uses eight Zepto policy documents:

```text
doc_01
doc_02
doc_03
doc_04
doc_05
doc_06
doc_07
doc_08
```

The documents are stored in:

```text
docs/
```

Each document is loaded and indexed during ingestion.

---

# 3. Document Ingestion

The ingestion pipeline is implemented in:

```text
ingest.py
```

The pipeline performs the following steps:

```text
Policy Documents
      ↓
Read .txt files
      ↓
Sentence Transformer
      ↓
Generate Embeddings
      ↓
Normalize Embeddings
      ↓
ChromaDB
```

The embedding model is:

```python
SentenceTransformer("all-MiniLM-L6-v2")
```

Embeddings are normalized before storage.

---

# 4. ChromaDB

ChromaDB is used as the persistent vector database.

The collection is:

```text
zepto_policies
```

The database directory is:

```text
chroma_db/
```

Cosine similarity is configured using:

```python
metadata={"hnsw:space": "cosine"}
```

Each stored document contains:

* Document ID
* Source filename
* Document text
* Embedding

---

# 5. Retrieval

For a user query, the query is converted into an embedding using the same Sentence Transformer model.

The embedding is normalized and searched against the ChromaDB collection.

The system retrieves:

```text
Top 3 relevant documents
```

using vector similarity.

The retrieved document IDs and document contents are then passed to the answer-generation stage.

---

# 6. LangGraph Architecture

The assistant uses LangGraph to orchestrate the workflow.

The graph contains three main nodes:

```text
classify_intent
retrieve_and_answer
direct_answer
```

The workflow is:

```text
                    User Query
                         |
                         v
                 classify_intent
                         |
                +--------+--------+
                |                 |
                v                 v
        policy_question     general_question
                |                 |
                v                 v
       retrieve_and_answer    direct_answer
                |                 |
                +--------+--------+
                         |
                         v
                       END
```

---

# 7. Intent Classification

The `classify_intent` node identifies whether the query is:

```text
policy_question
```

or:

```text
general_question
```

When `MOCK_LLM=1`, keyword-based classification is used for policy-related terms such as:

```text
delivery
return
refund
membership
tracking
cancel
gift card
support hours
```

When `MOCK_LLM=0`, the configured real LLM is used for intent classification.

---

# 8. Policy Question Flow

For a policy-related question:

```text
User Query
    ↓
classify_intent
    ↓
policy_question
    ↓
Query Embedding
    ↓
ChromaDB Retrieval
    ↓
Top 3 Documents
    ↓
Answer Generation
    ↓
Pydantic Validation
    ↓
Structured Response
```

The retrieval node uses:

```python
n_results=3
```

to retrieve the top three documents.

---

# 9. General Question Flow

For a general question:

```text
User Query
    ↓
classify_intent
    ↓
general_question
    ↓
direct_answer
    ↓
Structured Response
```

General questions do not go through the policy retrieval branch.

---

# 10. Grounded Answer Generation

The system prompt instructs the assistant to use only the retrieved Zepto policy context.

The prompt includes a negative constraint:

```text
Do not answer using information that is not present
in the provided context.
```

The prompt also instructs the model not to invent:

```text
Zepto policies
prices
timings
rules
```

This helps keep policy responses grounded in the retrieved documents.

---

# 11. Few-Shot Example

The prompt contains a few-shot example for a delivery-fee question.

Example:

```text
Question:
What is the delivery fee for orders below INR 149?

Context:
Orders below INR 149 incur a flat INR 25 delivery fee.
```

The expected structured response contains:

```text
answer
sources
confidence
```

---

# 12. Pydantic Structured Output

Responses are validated using the following schema:

```python
class AnswerResponse(BaseModel):
    answer: str
    sources: list[str]
    confidence: float
```

The confidence field is constrained between:

```text
0 and 1
```

The response is validated using:

```python
AnswerResponse.model_validate(...)
```

This ensures that the final response follows the expected structured format.

---

# 13. Validation and Retry

When the real LLM mode is enabled, the system validates the generated JSON response.

If the response fails validation:

```text
Attempt 1
   ↓
Validation
   ↓
Failure
   ↓
Corrective Prompt
   ↓
Attempt 2
   ↓
Validation
   ↓
Attempt 3
```

The system allows up to three attempts.

If structured-output validation still fails, an error response is returned with:

```text
confidence = 0.0
```

---

# 14. MOCK_LLM Mode

The default configuration enables:

```text
MOCK_LLM=1
```

This allows the project to run without requiring an external LLM API.

In mock mode:

* Intent classification uses deterministic keyword logic.
* Policy queries use ChromaDB retrieval.
* Retrieved context is used to construct a deterministic response.
* General questions return a deterministic response.
* Pydantic validation is still applied.

Therefore, the graded baseline can run completely offline for generation.

---

# 15. Real LLM Mode

The project also contains an optional real LLM integration.

When:

```text
MOCK_LLM=0
```

the application expects:

```text
GROQ_API_KEY
```

The default configured model is:

```text
llama-3.3-70b-versatile
```

The real LLM is used for:

* Intent classification
* Policy answer generation
* General question answering

The policy-answer prompt still requires grounded responses and structured JSON output.

---

# 16. FastAPI API

The application exposes a FastAPI endpoint:

```text
POST /ask
```

The endpoint accepts a user query and invokes the LangGraph workflow.

Conceptually:

```text
Client
  ↓
POST /ask
  ↓
LangGraph
  ↓
Intent Classification
  ↓
Retrieval / Direct Answer
  ↓
Pydantic Response
  ↓
JSON Response
```

---

# 17. Example Request

```json
{
  "query": "What is the delivery fee for orders below INR 149?"
}
```

The query is processed through the policy retrieval branch.

---

# 18. Example Response Structure

```json
{
  "answer": "Based on the retrieved context: ...",
  "sources": ["doc_01"],
  "confidence": 1.0
}
```

The exact answer depends on the retrieved policy context and execution mode.

---

# 19. Docker

The project includes:

```text
Dockerfile
```

The container installs the required Python dependencies and starts the FastAPI application using Uvicorn.

The configured application port is:

```text
7860
```

---

# 20. Verification

The default `MOCK_LLM` mode was tested with a policy question:

```text
What is the delivery fee for orders below INR 149?
```

The retrieval pipeline successfully retrieved `doc_01` as the top source along with related documents.

A general question was also tested and routed to the `direct_answer` branch without policy retrieval.

ChromaDB was verified with the eight indexed documents:

```text
doc_01
doc_02
doc_03
doc_04
doc_05
doc_06
doc_07
doc_08
```

---

# 21. Running the Ingestion Pipeline

From the `support_assistant` directory:

```bash
python ingest.py
```

This performs:

```text
Read Policy Documents
        ↓
Generate Embeddings
        ↓
Normalize Embeddings
        ↓
Store in ChromaDB
```

The script prints the number of indexed documents and their IDs.

---

# 22. Running the Assistant

The graph can be tested directly using:

```bash
python graph.py
```

This tests both:

```text
Policy Question
```

and:

```text
General Question
```

---

# 23. Running FastAPI

Start the API using:

```bash
uvicorn main:app --host 0.0.0.0 --port 7860
```

The API is then available through:

```text
POST /ask
```

---

# 24. Complete Architecture

The complete system can be summarized as:

```text
                 ZEPO POLICY DOCUMENTS
                         |
                         v
                    ingest.py
                         |
                         v
              Sentence Transformer
               all-MiniLM-L6-v2
                         |
                         v
                     Embeddings
                         |
                         v
                    ChromaDB
                         |
                         |
                    User Query
                         |
                         v
                  FastAPI /ask
                         |
                         v
                 classify_intent
                         |
              +----------+----------+
              |                     |
              v                     v
       policy_question       general_question
              |                     |
              v                     v
       Query Embedding        direct_answer
              |
              v
        ChromaDB Search
              |
              v
         Top 3 Documents
              |
              v
       Answer Generation
              |
              v
      Pydantic Validation
              |
              v
        Structured JSON
```

---

## Summary

The Zepto Support Assistant combines:

* Document ingestion
* Sentence Transformer embeddings
* ChromaDB vector retrieval
* Cosine similarity
* LangGraph orchestration
* Intent-based routing
* Grounded RAG
* Pydantic structured output
* Validation and retry logic
* FastAPI serving
* Docker containerization
* Offline `MOCK_LLM` execution
* Optional real LLM integration

The implementation is designed so that policy questions use retrieved Zepto policy context, while general questions follow a separate direct-answer branch.
