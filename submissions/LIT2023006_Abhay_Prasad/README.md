# Data Mining Assignment 2: Book Recommender on a Medallion Data Lake

**Student:** Abhay Prasad

**Roll number:** LIT2023006

**Dataset:** [Book-Crossing](https://cdn.freecodecamp.org/project-data/books/book-crossings.zip) (`BX-Users.csv`, `BX-Books.csv`, `BX-Book-Ratings.csv`)

## Run in Google Colab

1. Open [`Data_Lake_Assignment.ipynb` in Google Colab](https://colab.research.google.com/github/abhay9494/data-mining-assignment-2/blob/main/submissions/LIT2023006_Abhay_Prasad/Data_Lake_Assignment.ipynb).
2. Use a standard Python CPU runtime and choose **Runtime > Run all**. The whole notebook takes about a minute.
3. The notebook downloads the zip itself and checks its SHA-256, so no file upload or Google Drive mount is needed. It then creates `bronze/`, `silver/` and `gold/` in the working folder.

The notebook also runs in local Jupyter with Python 3.10+ and pandas, NumPy, SciPy, scikit-learn, PyArrow and Matplotlib. Run it from this folder.

## Structure

```text
submissions/LIT2023006_Abhay_Prasad/
├── Data_Lake_Assignment.ipynb     # executed notebook with all outputs
├── README.md
├── verification_summary.json     # key counts and results written by the notebook
├── bronze/                       # raw source tables + ingestion_timestamp (Parquet)
│   ├── books.parquet
│   ├── users.parquet
│   └── ratings.parquet
├── silver/                       # cleaned, typed, filtered tables (Parquet)
│   ├── books.parquet
│   ├── users.parquet
│   └── ratings.parquet
└── gold/                         # model-ready data
    ├── user_book_matrix.npz      # SciPy CSR sparse matrix, users x books, ratings 1-10
    ├── user_index.parquet        # matrix row -> user_id
    ├── book_lookup.parquet       # matrix column (book_idx) -> isbn, title, author, year
    ├── user_stats.parquet        # per user: num_ratings, avg_rating, age, country
    └── book_stats.parquet        # per book: num_ratings, avg_rating
```

The downloaded zip and the extracted CSV files go to `raw/`. This folder is git-ignored because the notebook re-creates it.

## Pipeline

**Step 1: Bronze.** The files are read with `sep=';'`, `encoding='latin-1'` and `escapechar='\\'`. Without the escape character, `BX-Books.csv` breaks at line 6,452 on titles such as `Of Mice and Men (Steinbeck \"Essentials\")`. As UTF-8 the file does not decode. Every column is kept as raw text with no missing-value conversion, so the literal `NULL` ages stay as they are. `ingestion_timestamp` is added and the tables are saved as Parquet, then read back to confirm they are identical.

**Step 2: EDA.** One finding and the Silver action for each item:

| Item | Finding | Silver action |
|---|---|---|
| Shape, types, missing | 271,379 books, 278,858 users and 1,149,780 ratings, all stored as text. No empty cells, but 110,762 ages are `NULL`, 343 authors are placeholders and 6.9% of locations contain `n/a` parts | Cast types, turn placeholders into nulls |
| Ratings | 716,109 (62.3%) are 0, which means an implicit interaction, not a score. Explicit ratings average 7.60 | Keep 1-10 only |
| Ages and years | Ages run from 0 to 244 (882 below 5, 366 above 100). 4,619 years are 0 and 11 are later than 2026 | Null the impossible values, fill ages with the median (32) |
| Ratings per user/book | The median user and the median book each have 1 explicit rating. Only 9.4% of users and 3.0% of books have 10 or more | Minimum-ratings filter |
| Sparsity | Only 0.0030% of user-book pairs have an explicit rating | Sparse matrix in Gold |

**Step 3: Silver.** The cleaning steps are:
- Remove exact duplicate rows and duplicate keys. This removes 315 ISBNs that duplicated others once trimmed and upper-cased.
- Remove ratings of 0.
- Set ages below 5 or above 100 to null, then fill all missing ages with the median (32) and add an `age_imputed` flag.
- Split `Location` into city, state and country, setting placeholder parts to null.
- Set publication year 0 and years after the ingestion year to null.
- Clean the text columns:
  - repair UTF-8 text that was decoded as Latin-1 (`ExupÃ©ry` becomes `Exupéry`)
  - convert HTML entities (`&amp;` becomes `&`)
  - remove extra spaces
  - convert all-capitals or all-lower-case text to title case
  - set placeholder authors to null
- Drop the image-URL columns.
- Keep only ratings whose book and user exist.

**Threshold:** at least **10 ratings per user and per book**. The filter repeats until it is stable, so every remaining user and book meets the minimum. Below 10, a book has too few co-raters for a meaningful cosine similarity. Above 10 almost nothing is left: 15/15 and 20/20 remove every rating. With 10/10, Silver keeps **41,469 ratings from 1,820 users on 2,030 books**, and 1.12% of the matrix is filled. Twelve automatic checks confirm the Silver tables.

**Step 4: Gold.** Gold holds a 1,820 × 2,030 CSR sparse rating matrix (0.34 MB, against 14.8 MB dense), the book lookup table and the user and book aggregates. Assertions confirm that Gold agrees with Silver.

**Step 5: Recommender (item-based collaborative filtering).**
- The matrix is transposed so that rows are books and columns are users.
- `NearestNeighbors(metric='cosine', algorithm='brute')` stores the 50 most similar books for every book. A direct `sklearn` `cosine_similarity` calculation confirms the values.
- `similar_books(title, n=10)` accepts an exact or partial title, ignoring case.
- `recommend(user_id, n=10)` scores each candidate as Σ similarity × the user's rating, over the books the user rated 8-10. It then removes books already read, including other editions of them, and returns the top `n` titles.

**Step 6: Demo and sanity check.** Users 182442, 29209 and 216466 are chosen at random (seed 42). For each one, the books they rated highly are printed next to the recommendations and a "because you liked" explanation. No recommendation has already been read.

In a held-out test on 500 users, `recommend()` puts the hidden highly rated book back in the top 10 for **12.8%** of users. The most-popular baseline manages **4.0%**, and a random pick about 0.5%.

Example: the books most similar to *Harry Potter and the Chamber of Secrets* are the other Harry Potter books (cosine similarity 0.44-0.70), followed by *The Hobbit*. The neighbours of *The Fellowship of the Ring* are the other Lord of the Rings books.

## Verification

The submitted notebook was executed from top to bottom in Jupyter on November 26, 2026, with no errors. It used Python 3.12 with Colab's library versions: pandas 2.2.2, NumPy 2.0.2, scikit-learn 1.6.1, SciPy 1.15.3, PyArrow 18.1.0 and Matplotlib 3.10.0. The same code was also run with pandas 3.0.2 and scikit-learn 1.8.0 and gave identical results.

All counts in the outputs are calculated at runtime, and `verification_summary.json` records them. Assertions check:

- Bronze and Silver read back exactly.
- The Silver quality rules all hold.
- The Gold matrix and aggregates agree with Silver.
- The neighbour similarities match a direct cosine-similarity calculation.
- Recommendations never include books the user has already read.

All submitted changes are inside this student's folder. The class README and other submissions are unchanged.
