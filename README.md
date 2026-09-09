# Zepto Data & AI Platform

## Overview

This project is an end-to-end AI/ML engineering capstone consisting of three connected modules:

1. Data Pipeline
2. Analytics Pipeline
3. GenAI Support Assistant

All three modules are contained in one public GitHub repository.

The project demonstrates data collection, data cleaning, relational database design, exploratory data analysis, machine learning, retrieval-augmented generation, workflow orchestration, API development, and local containerization.

## Project Structure

```text
zepto-data-ai-platform/
│
├── README.md
│
├── data_pipeline/
│   ├── scrape.py
│   ├── clean.py
│   ├── database.py
│   ├── queries.sql
│   ├── run_queries.py
│   ├── zepto.db
│   ├── README.md
│   └── output/
│       ├── scraped_books.csv
│       ├── cleaned_books.csv
│       └── query_results/
│           ├── query_1.csv
│           ├── query_2.csv
│           ├── query_3.csv
│           ├── query_4.csv
│           ├── query_5.csv
│           ├── query_6_join.csv
│           └── join_comparison.csv
│
├── analytics/
│   ├── eda.py
│   ├── modeling.py
│   ├── titanic.csv
│   ├── titanic_cleaned.csv
│   ├── README.md
│   └── artifacts/
│       ├── best_pipeline.joblib
│       ├── classification_results.csv
│       ├── imbalance_results.csv
│       ├── regression_results.csv
│       └── plots/
│
└── support_assistant/
    ├── docs/
    │   ├── doc_01.txt
    │   ├── doc_02.txt
    │   ├── doc_03.txt
    │   ├── doc_04.txt
    │   ├── doc_05.txt
    │   ├── doc_06.txt
    │   ├── doc_07.txt
    │   └── doc_08.txt
    ├── chroma_db/
    ├── ingest.py
    ├── graph.py
    ├── main.py
    ├── requirements.txt
    ├── Dockerfile
    ├── .dockerignore
    └── README.md