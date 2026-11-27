# Book Recommender on a Medallion Data Lake (Book-Crossing)

Data Mining Lab Assignment: Bronze → Silver → Gold → item-based collaborative filtering.

## Files
| File | Description |
|---|---|
| `recommender_medallion.ipynb` | Full notebook (Steps 1-6) with outputs and EDA findings. Runs top-to-bottom in Google Colab. |
| `recommender_medallion.html` | Same notebook as a read-only web page. |
| `bronze/` | Untouched raw copy of the three source tables (Parquet, with `ingestion_timestamp`). |
| `silver/` | Cleaned users, books, ratings (Parquet). |
| `gold/` | Sparse rating matrix (`.npz`), book lookup, index maps, user/book aggregates. |

The raw CSVs are not committed; the notebook downloads the zip and regenerates everything.

## How to run
Open the notebook in Colab → *Runtime → Run all*. It downloads the dataset, creates `bronze/ silver/ gold/`, and runs the whole pipeline (about 2-3 minutes).

## Key results
- Bronze: 278,858 users, 271,379 books, 1,149,780 ratings (raw, plus `ingestion_timestamp`).
- 62.3% of ratings are 0 (implicit); matrix sparsity is 99.997%.
- Silver threshold: users ≥ 5 ratings, books ≥ 10 ratings → 4,749 users, 3,060 books, 69,346 ratings.
- Model: cosine `NearestNeighbors` on the book × user sparse matrix; `recommend()` scores candidates by similarity × the user's rating, removes already-read books (any edition) and returns the top n.
