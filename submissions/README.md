# Data Mining Lab Assignment 2: Medallion Data Lake & Recommender System

## Overview
This repository contains the complete end-to-end implementation of a **Medallion Data Lake (Bronze -> Silver -> Gold)** and an **Item-Based Collaborative Filtering Recommender System** built on the **Book-Crossing** dataset.

The workflow is available as an executable, self-contained Google Colab Jupyter Notebook:
- **Notebook**: [`recommender_system_medallion.ipynb`](./recommender_system_medallion.ipynb)

---

## Architecture: Medallion Data Lake Pipeline

```
Raw CSV Files (Semicolon delimited, Latin-1 encoded)
        |
        v
[ Bronze Layer ] (bronze/*.parquet)
   - Untouched raw ingest with ISO UTC ingestion_timestamp
   - Ratings (1,149,780 rows), Users (278,858 rows), Books (271,360 rows)
        |
        v
[ Silver Layer ] (silver/*.parquet)
   - Deduplication across users, books, and ratings
   - Explicit rating isolation: filtered out 716,109 implicit '0' ratings (kept 1 to 10)
   - Age outlier trimming: set values < 5 or > 100 to null, imputed with median (32)
   - User Location parsing: split into standardized (city, state, country)
   - Publication Year bounds: converted to numeric; 0 and future years (> 2026) set to null
   - Text sanitation: stripped trailing whitespace, standardized casing, dropped unused image URLs
   - Referential integrity: enforced strict Foreign Key constraints between ratings, books, and users
   - Threshold filtering: retained users and books with >= 10 ratings (justified by EDA)
        |
        v
[ Gold Layer ] (gold/*)
   - Model-ready Compressed Sparse Row (CSR) Book-User Matrix (5,438 books x 6,318 users)
   - Book Metadata Lookup Store (book_id, title, author, year, publisher)
   - User-Level and Book-Level Aggregates (average rating and interaction counts)
        |
        v
[ Recommender Engine ]
   - NearestNeighbors model with cosine similarity metric
   - get_similar_books(book_title, n=10): returns top 10 most similar books by cosine similarity
   - recommend(user_id, n=10): computes rating-weighted similarity scores, filters already-read books
```

---

## Key EDA Insights & Actions

| EDA Metric | Observation in Bronze | Action Taken in Silver |
| :--- | :--- | :--- |
| **Shape & Nulls** | Ratings: 1,149,780 rows; Users: 278,858 rows (39.72% missing Age); Books: 271,360 rows (slight nulls). | Deduplicated all entities, dropped unneeded image URLs, and imputed missing ages with median (32). |
| **Ratings Distribution** | 716,109 ratings (62.28%) are `0` (implicit interactions without explicit scores). | Filtered out all ratings of `0`, retaining only explicit ratings in range $[1, 10]$. |
| **Age & Year Outliers** | User ages ranged from 0 to 244 (1,447 anomalies). Publication years had 4,618 zeros and 11 future years (up to 2050). | Set ages $< 5$ or $> 100$ to null then filled with median; set publication years $\le 0$ or $> 2026$ to null. |
| **Interactions per Entity** | Extreme heavy-tail: 55.45% of users and 51.52% of books have only 1 rating. >90% of users have $\le 12$ ratings; >95% of books have $\le 10$ ratings. | Enforced minimum activity threshold: kept users and books with $\ge 10$ ratings to eliminate cold-start noise. |
| **Matrix Sparsity** | Raw user-book matrix is **99.996793% sparse** (only ~0.0032% observed entries). | Threshold filtering compressed the matrix to 99.7463% sparsity, enabling memory-efficient CSR sparse storage. |

---

## Recommender System Formulation

### 1. Item-Based Cosine Similarity
Items are represented as sparse rating vectors across all users $\mathbf{r}_i \in \mathbb{R}^{U}$. The cosine similarity between book $i$ and book $j$ is:
$$\text{Sim}(i, j) = 1 - \text{Cosine Distance}(i, j) = \frac{\mathbf{r}_i \cdot \mathbf{r}_j}{\|\mathbf{r}_i\|_2 \|\mathbf{r}_j\|_2}$$

### 2. Rating-Weighted Recommendation Scoring
For a target user $u$, let $\text{Liked}(u)$ be their highly rated books ($r_{u, i} \ge 8$, or their top-rated items). For any candidate book $c$ not yet read by user $u$:
$$\text{Score}(c) = \sum_{i \in \text{Liked}(u)} \text{Sim}(c, i) \times r_{u, i}$$

---

## Verification & Sanity Check Highlights

### 1. Item Similarity Query: *"The Da Vinci Code"*
- **#1 Most Similar**: *"Angels & Demons"* by Dan Brown ($\text{Similarity} = 0.2085$)
- **#2**: *"Middlesex: A Novel"* by Jeffrey Eugenides ($\text{Similarity} = 0.1372$)
- **#3**: *"The Lovely Bones: A Novel"* by Alice Sebold ($\text{Similarity} = 0.1364$)

### 2. Personalized User Recommendations:
- **User 276925** (Top rated: *Politically Correct Bedtime Stories* [10/10], *The Da Vinci Code* [8/10]):
  - Recommended: *Once upon a More Enlightened Time: More Politically Correct Bedtime Stories* by James Finn Garner (#1), *Angels & Demons* by Dan Brown (#2).
- **User 278418** (Top rated: *Green Eggs and Ham* by Dr. Seuss [10/10], *Carolina Moon* by Nora Roberts [8/10]):
  - Recommended: *Heaven and Earth* by Nora Roberts (#1), *Fox in Socks* by Dr. Seuss (#2), *Jewels of the Sun* by Nora Roberts (#3).
- **User 114368** (Top rated: *Sacred Sins*, *Sweet Revenge*, *Public Secrets* by Nora Roberts [10/10]):
  - Recommended matching high-affinity romantic suspense and mystery fiction, with 100% of already-consumed books filtered out.

---

## How to Run
1. Open [`recommender_system_medallion.ipynb`](./recommender_system_medallion.ipynb) in **Google Colab** or Jupyter Notebook.
2. The initial setup cell automatically checks for local data and downloads `book-crossings.zip` if running in Colab.
3. Run all cells sequentially (`Runtime > Run all`).
