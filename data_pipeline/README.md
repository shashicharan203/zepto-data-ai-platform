# Data Pipeline

## Overview

This module builds a raw-to-relational data pipeline using book data from books.toscrape.com. The pipeline scrapes catalog data, cleans and transforms the fields, converts GBP prices to INR using the required fixed project rate, stores the data in a normalized SQLite database, and executes SQL and pandas queries.

## Data Source

The data is collected from books.toscrape.com using the requests and BeautifulSoup libraries.

The pipeline scrapes books from three categories:

- Travel
- Mystery
- Historical Fiction

The final dataset contains 69 books.

## Pipeline

The pipeline follows these stages:

Scraping → Cleaning → Currency Conversion → SQLite Loading → SQL Queries → Pandas Validation

## Files

### scrape.py

Scrapes book information including:

- title
- price
- star rating
- availability
- category

The script follows pagination for each selected category.

### clean.py

Transforms the scraped fields into usable data types:

- `price_gbp` is stored as a float.
- `rating` converts One–Five into integers from 1–5.
- `in_stock` converts availability text into a Boolean value.
- `price_inr` is calculated using the fixed conversion rate.

### database.py

Creates and populates the SQLite database using two normalized tables:

- `categories`
- `books`

The `books.category_id` column references `categories.category_id`.

### queries.sql

Contains SQL queries demonstrating SELECT, WHERE, ORDER BY, LIMIT, DISTINCT, IN, and JOIN.

### run_queries.py

Executes the SQL queries, saves their outputs, and reproduces the JOIN result using pandas `merge`.

## Currency Conversion

The project uses the required fixed baseline conversion:

**1 GBP = 105.50 INR**

Therefore:

`price_inr = price_gbp × 105.50`

No external currency API is used.

## Cleaning Decisions

Price values are cleaned by removing the GBP currency symbol and converting the remaining value to float.

Text ratings are mapped as follows:

- One → 1
- Two → 2
- Three → 3
- Four → 4
- Five → 5

Availability containing "In stock" is converted to `True`, while "Out of stock" is converted to `False`.

If numeric parsing fails, the median of the valid values is used for imputation. Rows with unparseable availability are dropped because the stock state cannot be reliably determined.

## Database Design

The database uses a normalized two-table structure.

### categories

- `category_id` — Primary Key
- `category_name` — Unique category name

### books

- `book_id` — Primary Key
- `title`
- `price_gbp`
- `price_inr`
- `rating`
- `in_stock`
- `category_id` — Foreign Key

The foreign key connects each book to its category without repeatedly storing the category name in every book record.

## SQL Queries

Six queries are included.

1. Filter books using `WHERE`.
2. Sort books using `ORDER BY`.
3. Return a limited number of rows using `LIMIT`.
4. Retrieve unique categories using `DISTINCT`.
5. Filter ratings using `IN`.
6. Join books and categories using `JOIN`.

Query outputs are saved under:

`output/query_results/`

## SQL and Pandas JOIN Validation

The JOIN query is executed using SQL and reproduced using pandas `pd.merge`.

Both results are compared programmatically.

The results are equivalent:

`Do both results match? True`

## How to Run

From the `data_pipeline` directory:

```bash
python scrape.py
python clean.py
python database.py
python run_queries.py