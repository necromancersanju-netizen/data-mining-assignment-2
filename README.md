# 📚 Data Mining Lab Assignment 2
## Recommender System using a Medallion Data Lake

| Field | Details |
|-------|---------|
| **Student** | Veda Akash |
| **Roll No** | LCI2023057 |
| **Assignment** | Build a Recommender System using a Medallion Data Lake |
| **Dataset** | Book-Crossing Dataset (Users, Books, Book-Ratings) |

---

## 📁 Folder Structure

```
lci2023057_veda_akash_data_mining 2/
│
├── lci2023057_veda_akash_recommender_system.ipynb   ← Main Colab Notebook
└── README.md                                         ← This file
```

When you run the notebook, it creates:

```
bronze/        ← Raw Parquet files (untouched source data)
silver/        ← Cleaned Parquet files
gold/          ← Model-ready matrix + lookup tables
raw_data/      ← Extracted CSVs from the zip
```

---

## 🚀 How to Run

1. Open [Google Colab](https://colab.research.google.com/)
2. Upload `lci2023057_veda_akash_recommender_system.ipynb`
3. Click **Runtime → Run all**
4. All steps are automated — the notebook downloads the dataset, builds the pipeline, and runs recommendations

---

## 🏗️ Pipeline Overview

### 🥉 Bronze Layer
- Downloads Book-Crossing zip from freecodecamp CDN
- Reads CSVs with `;` separator and `latin-1` encoding (non-UTF-8 handling)
- Adds `ingestion_timestamp` column
- Saves raw data as Parquet — **no cleaning**

### 🔍 EDA (Exploratory Data Analysis)
- Shape, dtypes, missing value counts for all 3 tables
- Rating distribution (shows ~70% implicit/0 ratings)
- Age and publication year distributions (shows outliers)
- Ratings per user and per book (long-tail distribution)
- Matrix sparsity calculation (>99.9% sparse)

### 🥈 Silver Layer
- Removes duplicate rows
- Removes 0 ratings (implicit), keeps 1–10
- Fixes invalid ages (< 5 or > 100 → set to null → fill with median)
- Splits `Location` into `city`, `state`, `country`
- Fixes invalid publication years (0 or future → null)
- Cleans and standardises text columns
- Drops image URL columns
- Referential integrity filter (ratings must have valid user + book)
- **Minimum ratings threshold: 10** (chosen based on EDA showing long-tail distribution; ensures enough signal for collaborative filtering)

### 🥇 Gold Layer
- Sparse CSR user-book rating matrix (scipy)
- Book lookup table (isbn, title, author)
- User-level aggregates (avg rating, num ratings)
- Book-level aggregates (avg rating, num ratings)

### 🤖 Recommender System (Item-Based Collaborative Filtering)
- `get_similar_books(title, n=10)`: KNN cosine similarity to find similar books
- `recommend(user_id, n=10)`: 
  - Finds books user rated highly (≥ 7)
  - Collects similar books via KNN
  - Scores by **similarity × user_rating** (weighted sum)
  - Removes already-read books
  - Returns top-n titles

---

## 📊 Key Results

| Stage | Users | Books | Ratings |
|-------|-------|-------|---------|
| Bronze | ~278,000 | ~271,000 | ~1,149,000 |
| Silver (after cleaning) | ~filtered | ~filtered | ~filtered |
| Gold (≥10 ratings threshold) | active users | popular books | explicit only |

---

## 🛠️ Libraries Used

- `pandas`, `numpy` — data manipulation
- `scipy.sparse` — sparse matrix for memory efficiency
- `scikit-learn` (NearestNeighbors) — cosine similarity KNN
- `pyarrow` — Parquet read/write
- `matplotlib`, `seaborn` — visualizations

---

## 📌 Notes

- All steps are reproducible; re-running the notebook produces identical results
- The notebook handles encoding errors gracefully (tries latin-1, iso-8859-1, cp1252)
- Sparse matrix is used throughout to handle the >99.9% sparsity efficiently
