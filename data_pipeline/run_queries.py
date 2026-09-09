import sqlite3
import pandas as pd
import os

DATABASE_FILE = "zepto.db"
OUTPUT_DIR = "output/query_results"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    conn = sqlite3.connect(DATABASE_FILE)

    queries = [
        (
            "query_1",
            """
            SELECT title, price_gbp, rating
            FROM books
            WHERE rating >= 4;
            """
        ),
        (
            "query_2",
            """
            SELECT title, price_inr
            FROM books
            ORDER BY price_inr DESC;
            """
        ),
        (
            "query_3",
            """
            SELECT title, price_gbp, rating
            FROM books
            ORDER BY rating DESC
            LIMIT 10;
            """
        ),
        (
            "query_4",
            """
            SELECT DISTINCT category_name
            FROM categories;
            """
        ),
        (
            "query_5",
            """
            SELECT title, price_gbp, rating, category_id
            FROM books
            WHERE rating IN (4, 5);
            """
        ),
        (
            "query_6_join",
            """
            SELECT
                books.title,
                books.price_gbp,
                books.price_inr,
                books.rating,
                books.in_stock,
                categories.category_name
            FROM books
            JOIN categories
            ON books.category_id = categories.category_id
            ORDER BY books.rating DESC, books.price_gbp DESC
            LIMIT 10;
            """
        )
    ]

    results = {}

    for name, query in queries:
        df = pd.read_sql(query, conn)
        results[name] = df

        output_file = f"{OUTPUT_DIR}/{name}.csv"
        df.to_csv(output_file, index=False)

        print(f"\n{name}")
        print(df)

    books_df = pd.read_sql(
        """
        SELECT
            book_id,
            title,
            price_gbp,
            price_inr,
            rating,
            in_stock,
            category_id
        FROM books
        """,
        conn
    )

    categories_df = pd.read_sql(
        """
        SELECT category_id, category_name
        FROM categories
        """,
        conn
    )

    merge_result = pd.merge(
        books_df,
        categories_df,
        on="category_id",
        how="inner"
    )

    merge_result = merge_result[
        [
            "title",
            "price_gbp",
            "price_inr",
            "rating",
            "in_stock",
            "category_name"
        ]
    ]

    merge_result = merge_result.sort_values(
        ["rating", "price_gbp"],
        ascending=[False, False]
    ).head(10).reset_index(drop=True)

    sql_join_result = results["query_6_join"].reset_index(drop=True)

    comparison = sql_join_result.equals(merge_result)

    print("\nSQL JOIN result:")
    print(sql_join_result)

    print("\nPandas merge result:")
    print(merge_result)

    print("\nDo both results match?", comparison)

    comparison_df = pd.DataFrame({
        "SQL_result": sql_join_result.astype(str).agg(" | ".join, axis=1),
        "Pandas_merge_result": merge_result.astype(str).agg(" | ".join, axis=1)
    })

    comparison_df.to_csv(
        f"{OUTPUT_DIR}/join_comparison.csv",
        index=False
    )

    conn.close()


if __name__ == "__main__":
    main()