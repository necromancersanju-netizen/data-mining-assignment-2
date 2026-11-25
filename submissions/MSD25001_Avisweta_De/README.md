# Data Mining Assignment 2: Book Recommender on a Medallion Data Lake

**Name:** Avisweta De  
**Roll No.:** MSD25001

Notebook (saved with all outputs): [`Data_Lake_Assignment.ipynb`](Data_Lake_Assignment.ipynb)

## Pipeline

| Step | What happens | Output |
|---|---|---|
| 1. Bronze | The three `;`-separated, Latin-1 CSVs are loaded as raw text (no cleaning), each with an `ingestion_timestamp`. A record count confirms nothing was lost. | `bronze/*.parquet` |
| 2. EDA | Shape, types, missing values, rating / age / year distributions, ratings per user and book, matrix sparsity. Each finding is written up with its planned Silver fix. | charts + findings |
| 3. Silver | Deduplicate; keep only explicit ratings (1–10); null ages < 5 or > 100 and fill them with the median (32); split `Location` into city / state / country; null year 0 and years after the 2004 crawl; repair text (HTML entities, broken accents, case); drop image URLs; enforce referential integrity; merge the editions of a book into one `book_id`; keep users and books with **≥ 10 ratings** (applied iteratively). | `silver/*.parquet` |
| 4. Gold | Sparse CSR user × book rating matrix plus its index maps, a book lookup (id, title, author, …), and user-level and book-level aggregates. | `gold/*.npz`, `gold/*.parquet` |
| 5. Model | Book × user matrix → `NearestNeighbors(metric="cosine")`. `similar_books(title)` and `recommend(user_id, n=10)` score candidates by Σ similarity × the user's rating, skip books already read, and fall back to popular books for cold-start users. | functions |
| 6. Demo | Recommendations for three users, shown next to the books they rated highly, plus a cold-start example and a leave-one-out hit-rate test. | tables |

## Results

* Cleaning funnel: 1,149,780 raw ratings → 433,671 explicit → **62,836** model ratings. The model covers **2,472 users × 2,632 books** at 0.97% density, vs 0.0015% in the raw data.
* Leave-one-out test, hidden liked book found in the top 10: **item-CF 13.5%**, popularity 3.8%, random 0.4%.
* Example: *Harry Potter and the Sorcerer's Stone* → the other Harry Potter books, then *The Fellowship of the Ring*, *The Two Towers*, …

## How to run

1. Open the notebook in Google Colab (or Jupyter).
2. Either upload `BX-Users.csv`, `BX-Books.csv` and `BX-Book-Ratings.csv` to the working folder, or let the first cells download them from `https://cdn.freecodecamp.org/project-data/books/book-crossings.zip`.
3. Run all cells (about 1 minute). The notebook creates `bronze/`, `silver/` and `gold/` next to itself.

Requirements: `pandas`, `numpy`, `pyarrow`, `scipy`, `scikit-learn`, `matplotlib` (all preinstalled in Colab).

## Files in this folder

| Path | Content |
|---|---|
| `Data_Lake_Assignment.ipynb` | the full pipeline and recommender, executed |
| `silver/` | cleaned users, books and ratings (Parquet) |
| `gold/` | sparse user × book matrix (`.npz`), index maps, book lookup, user/book aggregates |

`bronze/` (35 MB, a raw copy of the CSVs) is not committed. The notebook rebuilds it, and the silver and gold layers, from the source files.
