import requests
from bs4 import BeautifulSoup
import pandas as pd


BASE_URL = "https://books.toscrape.com/"
CATEGORY_URL = "https://books.toscrape.com/catalogue/category/books/"


def get_soup(url):
    response = requests.get(url, timeout=10)
    response.raise_for_status()

    return BeautifulSoup(response.text, "html.parser")


def get_categories():
    soup = get_soup(BASE_URL)

    categories = []

    category_list = soup.select("div.side_categories ul li ul li a")

    for link in category_list:
        category_name = link.get_text(strip=True)
        category_url = BASE_URL + link["href"]

        categories.append({
            "category": category_name,
            "url": category_url
        })

    return categories


def scrape_category(category_name, category_url):
    books = []
    page_url = category_url

    while page_url:

        soup = get_soup(page_url)

        book_cards = soup.select("article.product_pod")

        for book in book_cards:

            title = book.h3.a["title"]

            price = book.select_one(".price_color").get_text(strip=True)

            rating = book.select_one("p.star-rating")["class"][1]

            availability = book.select_one(".availability").get_text(
                " ", strip=True
            )

            books.append({
                "title": title,
                "price": price,
                "star_rating": rating,
                "availability": availability,
                "category": category_name
            })

        next_button = soup.select_one("li.next a")

        if next_button:
            next_url = next_button["href"]

            if page_url.endswith("/"):
                page_url = page_url + next_url
            else:
                page_url = page_url.rsplit("/", 1)[0] + "/" + next_url
        else:
            page_url = None

    return books


def main():

    categories = get_categories()

    selected_categories = categories[:3]

    all_books = []

    for category in selected_categories:

        print(f"Scraping category: {category['category']}")

        books = scrape_category(
            category["category"],
            category["url"]
        )

        all_books.extend(books)

        print(f"Books collected: {len(books)}")

    df = pd.DataFrame(all_books)

    print("\nTotal books scraped:", len(df))
    print("Categories:", df["category"].nunique())

    output_file = "output/scraped_books.csv"

    df.to_csv(output_file, index=False)

    print(f"\nSaved data to: {output_file}")


if __name__ == "__main__":
    main()