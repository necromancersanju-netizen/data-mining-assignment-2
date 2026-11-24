# Book Recommender using a Medallion Data Lake

**Name:** Your Name
**Roll No:** Your Roll Number
**Notebook:** `2301234_SanjuKumar_medallion_recommender.ipynb`

## Pipeline
- **Bronze:** raw Book-Crossing CSVs (semicolon-separated, latin-1) saved as Parquet with an `ingestion_timestamp` column. No cleaning.
- **Silver:** removed duplicates and 0 ratings, fixed ages and publication years, split location into city/state/country, cleaned text, enforced referential integrity, and kept users and books with at least 10 ratings.
- **Gold:** sparse user-book rating matrix, book lookup table, and user-level and book-level aggregates.

## Recommender
Item-based collaborative filtering using cosine similarity (`NearestNeighbors`) on the sparse matrix.
- `similar_books(title)` returns the 10 most similar books.
- `recommend(user_id, n=10)` returns the top n books for a user, excluding books already read.

## Results
The demo for 3 users is in the notebook, with each user's highly rated books next to the recommendations.
