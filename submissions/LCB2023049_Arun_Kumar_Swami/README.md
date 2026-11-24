# Data Mining Assignment 2: Book Recommender System (Medallion Data Lake)

## Folder Contents
- `Data_Mining_Assignment_2.ipynb`: Full pipeline (Bronze → Silver → Gold → Item-Based CF Recommender & Sanity Check).
- `silver/`: Cleaned Parquet tables (`users`, `books`, `ratings` filtered to ≥ 10 explicit ratings).
- `gold/`: Sparse user-book rating matrix (`.npz`), book lookup table, index mappings, and user/book aggregates.
### Step 2: Bronze EDA Findings & Silver Plan
* **Shape, data types, and missing values:** `Users` (278,858 rows) has 110,762 missing `Age` values, `Books` (~271,360 rows) has 2 missing authors, 2 missing publishers, and 3 missing `Image-URL-L` values, and `Ratings` (1,149,780 rows) has no missing values; in Silver, we will drop the unneeded image-URL columns, clean text formatting, and handle missing values.
* **Distribution of ratings:** 716,109 ratings (62.28%) are `0` (implicit/no explicit rating); in Silver, we will remove all `0` ratings and keep only explicit ratings from 1 to 10.
* **User ages and publication years:** There are 1,248 invalid ages (< 5 or > 100, ranging from 0 to 244) and 4,629 invalid publication years (0 or > 2026, up to 2050); in Silver, we will set invalid ages to null and fill all null ages with the median (32), and set invalid publication years to null.
* **Ratings per user and per book:** The data is heavily skewed—the median user and median book each have only 1 rating (75% of users have ≤ 4 ratings and 75% of books have ≤ 2); in Silver, we will filter to keep only users and books with at least 10 explicit ratings so recommendations are reliable.
* **Matrix sparsity:** Across 105,283 unique users and 340,556 unique books, matrix density is only 0.003207% (99.996793% sparse); in Gold, we will store the user-book matrix as a `scipy.sparse` CSR matrix.