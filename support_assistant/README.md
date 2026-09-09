# Zepto Support Assistant

## Overview

This module implements a small Retrieval-Augmented Generation (RAG) support assistant for Zepto policies.

The application:

- Loads 8 Zepto policy documents
- Embeds the documents using `all-MiniLM-L6-v2`
- Stores embeddings in ChromaDB
- Uses LangGraph for intent routing
- Retrieves relevant policy context for policy questions
- Uses deterministic mock logic by default
- Validates responses using Pydantic
- Exposes the assistant through a FastAPI API
- Includes a Dockerfile for local containerization

The graded baseline runs completely offline for LLM generation.

## Project Structure

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
├── ingest.py
├── graph.py
├── main.py
├── requirements.txt
├── Dockerfile
├── .dockerignore
└── README.md

## Verification

The default MOCK_LLM mode was tested successfully.

The policy query "What is the delivery fee for orders below INR 149?" successfully retrieved `doc_01` along with the top related documents from ChromaDB.

A general query was also tested and correctly routed to the direct-answer branch without retrieval.
