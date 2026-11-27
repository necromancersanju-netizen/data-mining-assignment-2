# Book Recommender System using a Medallion Data Lake

Data Mining Lab Assignment. Dataset: Book-Crossing (Users, Books, Book-Ratings).

## How to run

1. Open `Recommender_System.ipynb` in Google Colab.
2. Choose **Runtime > Run all**.

The notebook downloads the data by itself. If the download fails, put `book-crossings.zip`
in a folder named `raw/` and run all cells again.

To run on your computer: `pip install -r requirements.txt jupyter`, then open the notebook.

## Layers

| Layer | What it holds |
|---|---|
| `bronze/` | The three raw tables as Parquet, with `ingestion_timestamp`. Nothing is cleaned. |
| `silver/` | Cleaned users, books and ratings. |
| `gold/` | Sparse user-book matrix, book lookup table, user and book aggregates. |

These folders are created when the notebook runs.

## Main choices

- Ratings of 0 are removed. Ratings 1 to 10 are kept.
- Ages below 5 or above 100 are set to null, then filled with the median age (32).
- ISBNs with the same title and author are treated as one book, so editions share their ratings.
- Each user and book must have at least **5** ratings. A limit of 10 left only 2,441 books.
- Recommender: item-based collaborative filtering with cosine similarity.
  The score of a book is the sum of (similarity x user's rating) over the books the user rated 8 or more.

## Result

| Item | Value |
|---|---|
| Users | 7,563 |
| Books | 9,352 |
| Ratings | 130,854 |
| Matrix sparsity | 99.815% |

## Limits

- The matrix is very empty, so similarity values are small.
- Users and books below the limit get no recommendations.
- The same book written with a different title can appear as a separate book.
