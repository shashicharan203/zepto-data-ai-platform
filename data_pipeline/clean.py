import pandas as pd
import os

INPUT_FILE = "output/scraped_books.csv"
OUTPUT_FILE = "output/cleaned_books.csv"

GBP_TO_INR = 105.50


def clean_price(price):
    try:
        price = str(price).strip()
        price = price.replace("Â£", "").replace("£", "")
        return float(price)
    except (ValueError, TypeError):
        return None


def convert_rating(rating):
    rating_map = {
        "One": 1,
        "Two": 2,
        "Three": 3,
        "Four": 4,
        "Five": 5
    }

    return rating_map.get(str(rating).strip())


def convert_stock(availability):
    availability = str(availability).strip().lower()

    if "in stock" in availability:
        return True

    if "out of stock" in availability:
        return False

    return None


def main():
    df = pd.read_csv(INPUT_FILE, encoding="utf-8-sig")

    print("Original rows:", len(df))

    df["price_gbp"] = df["price"].apply(clean_price)
    df["rating"] = df["star_rating"].apply(convert_rating)
    df["in_stock"] = df["availability"].apply(convert_stock)

    invalid_prices = df["price_gbp"].isna().sum()
    invalid_ratings = df["rating"].isna().sum()
    invalid_stock = df["in_stock"].isna().sum()

    print("Invalid prices:", invalid_prices)
    print("Invalid ratings:", invalid_ratings)
    print("Invalid availability:", invalid_stock)

    if invalid_prices > 0:
        df["price_gbp"] = df["price_gbp"].fillna(df["price_gbp"].median())

    if invalid_ratings > 0:
        df["rating"] = df["rating"].fillna(round(df["rating"].median()))

    df = df.dropna(subset=["in_stock"])

    df["price_gbp"] = df["price_gbp"].astype(float)
    df["rating"] = df["rating"].astype(int)
    df["in_stock"] = df["in_stock"].astype(bool)

    df["price_inr"] = df["price_gbp"] * GBP_TO_INR

    df = df[
        [
            "title",
            "price_gbp",
            "price_inr",
            "rating",
            "in_stock",
            "category"
        ]
    ]

    os.makedirs("output", exist_ok=True)

    df.to_csv(OUTPUT_FILE, index=False)

    print("\nCleaning completed successfully!")
    print("Final rows:", len(df))
    print("Saved to:", OUTPUT_FILE)

    print("\nData types:")
    print(df.dtypes)

    print("\nFirst 5 rows:")
    print(df.head())


if __name__ == "__main__":
    main()