# Book Recommender (Medallion Data Lake)

A book recommender system built on the Book-Crossing dataset for the Data Mining Lab Assignment.

## What it does

- **Bronze:** loads the raw data as it is and saves it as Parquet.
- **Silver:** cleans the data (removes duplicates and 0 ratings, fixes ages and years, splits location, keeps users and books with at least 10 ratings).
- **Gold:** builds a sparse user-book rating matrix, a book lookup table, and rating averages and counts.
- **Recommender:** item-based collaborative filtering using cosine similarity.
  - `similar_books(title)` gives the 10 most similar books.
  - `recommend(user_id, n=10)` gives book suggestions for a user.

## How to run

1. Open `Book_Recommender_Medallion.ipynb` in Google Colab.
2. Click **Runtime → Run all**. The dataset downloads automatically.

**Author:** Utkarsh
