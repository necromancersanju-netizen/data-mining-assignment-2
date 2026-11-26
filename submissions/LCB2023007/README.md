# Data Mining Lab Assignment 2 — Book Recommender System

**Student:** Akshay Kumar  
**Roll number:** LCB2023007

## Overview
This notebook implements a Bronze–Silver–Gold medallion data lake using the Book-Crossing dataset, followed by item-based collaborative filtering with cosine similarity on a SciPy CSR sparse rating matrix.

## Run in Google Colab
1. Open `Book_Recommender_Assignment_LCB2023007.ipynb` in Colab.
2. Obtain the [Book-Crossing ZIP](https://cdn.freecodecamp.org/project-data/books/book-crossings.zip) and upload `book-crossings.zip` using the Files panel to `/content/book-crossings.zip`. If the link is blocked, use the supplied local dataset ZIP.
3. Select Runtime > Run all. The notebook creates `bronze/`, `silver/`, and `gold/` automatically.

Dependencies: pandas, NumPy, PyArrow, SciPy, scikit-learn and Matplotlib.

## Pipeline
- Bronze: parse the semicolon-separated Latin-1 files, preserve source records, add ingestion timestamps and store Parquet.
- Silver: deduplicate, retain explicit ratings (1–10), fix ages and publication years, split locations, clean text, validate references and iteratively enforce a minimum of 10 ratings per user and book.
- Gold: store a CSR user-book matrix, lookup tables, user/book aggregates and seen-history index.
- Model: fit cosine nearest neighbors on the transposed matrix; rank unseen ISBNs by summed similarity weighted by the user's ratings. Highly rated books have ratings of at least 8, with a fallback for users without such ratings.

## Executed results
| Metric | Value |
|---|---:|
| Users | 1,820 |
| Books | 2,030 |
| Explicit ratings | 41,469 |
| Matrix sparsity | 98.88% |
| CSR memory | 0.34 MB |

The notebook demonstrates 10 recommendations each for users 243, 254 and 638. All programmed sanity checks passed. Exclusion is based on recorded ISBN interactions and exact normalized titles; alternate editions with different titles may still appear. No held-out accuracy evaluation is included.

The dataset and generated lake files are not included because the notebook regenerates them.
