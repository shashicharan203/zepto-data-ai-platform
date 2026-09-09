SELECT title, price_gbp, rating
FROM books
WHERE rating >= 4;

SELECT title, price_inr
FROM books
ORDER BY price_inr DESC;

SELECT title, price_gbp, rating
FROM books
ORDER BY rating DESC
LIMIT 10;

SELECT DISTINCT category_name
FROM categories;

SELECT title, price_gbp, rating, category_id
FROM books
WHERE rating IN (4, 5);

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