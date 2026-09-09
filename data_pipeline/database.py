import sqlite3
import pandas as pd
import os

INPUT_FILE = "output/cleaned_books.csv"
DATABASE_FILE = "zepto.db"


def create_tables(conn):
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS categories (
            category_id INTEGER PRIMARY KEY AUTOINCREMENT,
            category_name TEXT UNIQUE NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS books (
            book_id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            price_gbp REAL NOT NULL,
            price_inr REAL NOT NULL,
            rating INTEGER NOT NULL,
            in_stock INTEGER NOT NULL,
            category_id INTEGER NOT NULL,
            FOREIGN KEY (category_id) REFERENCES categories(category_id)
        )
    """)

    conn.commit()


def insert_categories(conn, df):
    categories = df["category"].drop_duplicates()

    for category in categories:
        conn.execute(
            """
            INSERT OR IGNORE INTO categories (category_name)
            VALUES (?)
            """,
            (category,)
        )

    conn.commit()


def insert_books(conn, df):
    category_map = pd.read_sql(
        "SELECT category_id, category_name FROM categories",
        conn
    )

    category_map = dict(
        zip(
            category_map["category_name"],
            category_map["category_id"]
        )
    )

    for _, row in df.iterrows():
        conn.execute(
            """
            INSERT INTO books
            (title, price_gbp, price_inr, rating, in_stock, category_id)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                row["title"],
                row["price_gbp"],
                row["price_inr"],
                row["rating"],
                int(row["in_stock"]),
                category_map[row["category"]]
            )
        )

    conn.commit()


def main():
    df = pd.read_csv(INPUT_FILE)

    if os.path.exists(DATABASE_FILE):
        os.remove(DATABASE_FILE)

    conn = sqlite3.connect(DATABASE_FILE)

    conn.execute("PRAGMA foreign_keys = ON")

    create_tables(conn)
    insert_categories(conn, df)
    insert_books(conn, df)

    category_count = pd.read_sql(
        "SELECT COUNT(*) AS count FROM categories",
        conn
    )

    book_count = pd.read_sql(
        "SELECT COUNT(*) AS count FROM books",
        conn
    )

    print("Categories inserted:", category_count.iloc[0]["count"])
    print("Books inserted:", book_count.iloc[0]["count"])

    print("\nCategories:")
    print(
        pd.read_sql(
            "SELECT * FROM categories",
            conn
        )
    )

    print("\nFirst 5 books:")
    print(
        pd.read_sql(
            "SELECT * FROM books LIMIT 5",
            conn
        )
    )

    conn.close()

    print("\nDatabase created successfully:", DATABASE_FILE)


if __name__ == "__main__":
    main()