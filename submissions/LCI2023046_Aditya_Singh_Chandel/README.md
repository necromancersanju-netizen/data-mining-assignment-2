# Medallion Data Lake & Item-Based Book Recommender System
**Data Mining Lab Assignment 2**

**Author:** Aditya Singh Chandel  
**Roll Number:** LCI2023046  
**Course:** Data Mining  
**Dataset:** Book-Crossing (Users, Books, Book-Ratings)  
**Submission Repository:** [data-mining-assignment-2](https://github.com/necromancersanju-netizen/data-mining-assignment-2)  

---

## 1. Project Architecture & Medallion Pipeline Overview

This project implements an enterprise-grade **Medallion Data Lakehouse Architecture** that ingests, cleans, conformedly models, and computes recommendations on the Book-Crossing dataset.

```
┌────────────────────────────────┐
│      Raw Source Data           │  (BX-Users, BX-Books, BX-Book-Ratings)
│   (Semicolon, Latin-1, Messy)  │
└───────────────┬────────────────┘
                │
                ▼
┌────────────────────────────────┐
│         BRONZE LAYER           │  • Exact untouched copy of source
│  (System of Record / Parquet)  │  • Appended UTC ingestion_timestamp
└───────────────┬────────────────┘  • Preserved raw strings & anomalies
                │
                ▼
┌────────────────────────────────┐
│         SILVER LAYER           │  • Deduplicated users, books, ratings
│ (Cleaned Enterprise Conformed) │  • Discarded 0 ratings (retained 1-10)
└───────────────┬────────────────┘  • Fixed ages (imputed median 32)
                │                   • Split location into city, state, country
                │                   • Enforced FK referential integrity
                │                   • Pruned cold-start noise (k >= 10)
                ▼
┌────────────────────────────────┐
│          GOLD LAYER            │  • book_lookup.parquet (clean metadata)
│  (Curated Model-Ready Assets)  │  • user_aggregates & book_aggregates
└───────────────┬────────────────┘  • Compressed Sparse Row (CSR) matrix
                │
                ▼
┌────────────────────────────────┐
│       RECOMMENDER ENGINE       │  • Item-Based Collaborative Filtering
│  (Cosine Nearest Neighbors)    │  • Rating-weighted similarity ranking
└────────────────────────────────┘  • Filtered previously read books
```

---

## 2. Quantitative EDA Findings & Silver Remediations

| EDA Dimension | Bronze Finding | Silver Layer Remediation |
|---|---|---|
| **1. Shape & Null Values** | `Users`: 278,858 rows (39.72% null Age); `Books`: 271,379 rows (minor nulls, unneeded URLs); `Ratings`: 1,149,780 rows. | Parsed numeric IDs, imputed missing ages, pruned unneeded image URL columns, and enforced referential integrity. |
| **2. Rating Distribution** | 716,109 ratings (62.28%) are 0 (implicit interactions); explicit ratings (1–10) average 7.60 with median 8.0. | Discarded all 0 ratings, training the collaborative filter strictly on explicit ratings (1–10). |
| **3. Age & Year Anomalies** | User ages contained 882 values < 5 and 366 values > 100; publication years contained 4,619 zeros and 11 future years (> 2026). | Replaced ages < 5 or > 100 with NaN and imputed using median valid age (32); set invalid publication years to null. |
| **4. Interaction Long-Tail** | Extreme power-law sparsity: over 50% of users and books have only 1 rating; only 9.43% of users and 3.04% of books have 10+ ratings. | Filtered users and books to those with at least 10 ratings ($k \ge 10$) to eliminate cold-start noise and guarantee overlap. |
| **5. Matrix Sparsity** | Total potential pairs: 35.85 billion. Observed pairs: 1.15 million. Matrix density was 0.0032% (99.9968% sparse). | Pruned low-support entities, elevating matrix density to 0.2538% (an 80x density boost) for reliable cosine neighborhoods. |

---

## 3. Recommender Engine: Item-Based Collaborative Filtering

### Similarity Metric
Cosine similarity is computed on sparse book interaction vectors $\vec{r}_i \in \mathbb{R}^M$ across all users:
$$\text{sim}(i, j) = \frac{\vec{r}_i \cdot \vec{r}_j}{\|\vec{r}_i\|_2 \|\vec{r}_j\|_2}$$

### Personalized Scoring
For any target user $u$, candidate books $j \notin \text{Read}(u)$ are scored by weighting candidate item similarities with the user's historical ratings:
$$\text{Score}(u, j) = \sum_{i \in \mathcal{H}_u} \text{sim}(i, j) \cdot r_{u, i}$$

---

## 4. Personalization & Sanity Check Highlights

### Demonstration on User 139913 (Crime & Mystery Enthusiast):
- **User's Favorite Books (Rated 10/10):**
  - *L Is for Lawless* by Sue Grafton
  - *The Third Victim* by Lisa Gardner
  - *The Notebook* by Nicholas Sparks
- **Engine's Top Recommendations:**
  1. *E Is for Evidence: A Kinsey Millhone Mystery* by Sue Grafton (Sim: 0.528, Score: 28.41)
  2. *I Is for Innocent* by Sue Grafton (Sim: 0.471, Score: 28.39)
  3. *K Is for Killer* by Sue Grafton (Sim: 0.361, Score: 23.07)
  4. *A Is for Alibi* by Sue Grafton (Sim: 0.317, Score: 16.12)
- **Sanity Verification:** The system successfully recovered the entire Kinsey Millhone detective sequence based purely on community co-rating patterns, achieving flawless domain and series continuity without needing genre metadata.

---

## 5. Directory Structure of Submission

```
submissions/LCI2023046_Aditya_Singh_Chandel/
├── medallion_book_recommender.ipynb  # Primary self-contained execution notebook
├── README.md                         # Complete methodology & results documentation
├── bronze/
│   ├── users.parquet                 # Untouched raw users + ingestion_timestamp
│   ├── books.parquet                 # Untouched raw books + ingestion_timestamp
│   └── ratings.parquet               # Untouched raw ratings + ingestion_timestamp
├── silver/
│   ├── users_silver.parquet          # Cleaned, conformed users with imputed ages & parsed locations
│   ├── books_silver.parquet          # Cleaned books without image URLs & with valid publication years
│   └── ratings_silver.parquet        # Cleaned explicit ratings (1-10) with k >= 10 filter
└── gold/
    ├── book_lookup.parquet           # Clean metadata lookup (ISBN, Title, Author, Year, Publisher)
    ├── user_aggregates.parquet       # User-level rating counts and averages
    ├── book_aggregates.parquet       # Book-level rating counts and averages
    ├── book_user_matrix.npz          # Scipy CSR matrix (5,440 books x 6,319 users)
    ├── book_index_mapping.parquet    # Fast ISBN to row index dictionary
    └── user_index_mapping.parquet    # Fast User-ID to column index dictionary
```

---

## 6. How to Run in Google Colab

1. Open [Google Colab](https://colab.research.google.com/).
2. Select **Upload Notebook** and choose `medallion_book_recommender.ipynb`.
3. Click **Runtime** > **Run all**.
4. The notebook autonomously creates the directory structure, downloads the Book-Crossing dataset, executes the Bronze-Silver-Gold pipeline, trains the collaborative filter, and displays the personalized recommendation demo.
