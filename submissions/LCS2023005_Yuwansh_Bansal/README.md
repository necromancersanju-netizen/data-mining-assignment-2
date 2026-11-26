# Book Recommender System using a Medallion Data Lake

**Name:** Yuwansh Bansal  
**Roll No:** LCS2023005

Data Mining Lab Assignment: Book-Crossing dataset → Bronze → Silver → Gold → Item-based Collaborative Filtering.

**Notebook:** `LCS2023005_Yuwansh_Bansal.ipynb` (saved with all outputs; open in Google Colab and choose *Runtime → Run all* to reproduce. It downloads the dataset itself).

## Pipeline

| Layer | Contents |
|---|---|
| **Bronze** | The 3 raw CSVs (`;`-separated, Latin-1, `\"`-escaped quotes) loaded as-is, plus `ingestion_timestamp`. Saved as Parquet. |
| **Silver** | Duplicates removed · ratings of 0 dropped (1–10 kept) · ages < 5 or > 100 set to null, then filled with the median (32) · `Location` split into city/state/country · years of 0 or after 2004 set to null · text trimmed, HTML entities decoded and Title-Cased · image-URL columns dropped · only ratings with a valid user and book kept · **10-core filter** (every user and book has ≥ 10 ratings) |
| **Gold** | Sparse CSR user×book matrix (`.npz`) with row/column index maps · book lookup (isbn, title, author) · user and book aggregates (average rating, number of ratings) |

## Key EDA numbers

* 1,149,780 ratings, of which **62.3% are 0** (implicit, not a score)
* Median user and median book have **1** explicit rating; one user has 8,524
* Explicit matrix sparsity: **99.997%** (77,805 users × 185,973 books)
* After cleaning + 10-core: **41,469 ratings · 1,820 users · 2,030 books** (density 1.1%)

## Recommender

* Books × users matrix → `NearestNeighbors(metric='cosine', algorithm='brute')`
* `similar_books(title)` returns the 10 most similar books (e.g. *Harry Potter 1* → Harry Potter 2, 3, 4, 5)
* `recommend(user_id, n=10)`: the user's books rated ≥ 8 → their neighbours → score = Σ similarity × user's rating → books already read removed → top n, with a "because you liked" explanation for each
* The demo covers 3 users (heavy, medium and light readers), with the books they rated highly shown next to their recommendations

## Folder contents

* `LCS2023005_Yuwansh_Bansal.ipynb`: the full notebook, run in Google Colab with all outputs
* `bronze/`, `silver/`, `gold/`: the data-lake layers the notebook produces (Parquet, plus the sparse `.npz` rating matrix)
