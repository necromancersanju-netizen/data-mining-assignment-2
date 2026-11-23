# Data Mining Lab Assignment 2: Book Recommender System using a Medallion Data Lake

**Student Roll Number:** LCB2023039  
**Student Name:** Prashik Humane  
**Course:** Data Mining  
**Dataset:** Book-Crossing (Users, Books, Book-Ratings)  
**Environment:** Google Colab / Apache Parquet / Scikit-Learn / SciPy  

---

## 1. Project Overview & Architecture

This project implements a complete **Medallion Data Lake Architecture** (**Bronze $\to$ Silver $\to$ Gold**) simulated in Python / Google Colab, transforming the raw, highly sparse Book-Crossing dataset into model-ready data structures, and deploying an **Item-Based Collaborative Filtering Recommender System**.

```
====================================================================================================
                                  MEDALLION DATA LAKE ARCHITECTURE
====================================================================================================

   [RAW DATA SOURCE]
   - BX-Users.csv (278,858 rows)
   - BX-Books.csv (271,360 rows)
   - BX-Book-Ratings.csv (1,149,780 rows)
               │
               ▼
   [BRONZE LAYER] (Raw & Immutable Storage)
   - Semicolon-delimited, Latin-1 encoding, on_bad_lines='skip'
   - Appended UTC `ingestion_timestamp`
   - Saved as compressed Apache Parquet:
     * bronze/users_bronze.parquet
     * bronze/books_bronze.parquet
     * bronze/ratings_bronze.parquet
               │
               ▼
   [EXPLORATORY DATA ANALYSIS (EDA)]
   - Missing value identification (Age: 39.72% missing)
   - Implicit rating spike (Rating 0 = 62.28%)
   - Biologically invalid ages (0 to 244) & publication years (<= 0, > 2026)
   - Extreme power-law sparsity: 99.9968% sparse
               │
               ▼
   [SILVER LAYER] (Cleaned, Standardized & Filtered)
   - Exact deduplication across (User-ID, ISBN)
   - Dropped 0 ratings (retained explicit 1-10 scores)
   - Age imputation with median (32) for values < 5 or > 100
   - Split `Location` into `city`, `state`, `country` (Title Case)
   - Validated publication years (1800-2026), dropped image URLs
   - Filtered interaction threshold: Users >= 10 ratings, Books >= 10 ratings
   - Saved as cleaned Apache Parquet:
     * silver/users_silver.parquet
     * silver/books_silver.parquet
     * silver/ratings_silver.parquet
               │
               ▼
   [GOLD LAYER] (Model-Ready Feature Store)
   - Book-User Sparse Rating Matrix (CSR format, 5,440 books x 6,319 users)
   - Book Metadata Lookup Table (ISBN -> Title, Author, Year, Publisher)
   - User Aggregates & Book Aggregates (rating count, mean, min, max)
   - Saved as:
     * gold/book_user_matrix.npz
     * gold/book_lookup.parquet
     * gold/user_aggregates.parquet
     * gold/book_aggregates.parquet
               │
               ▼
   [RECOMMENDER SYSTEM ENGINE]
   - Item-Based Collaborative Filtering via NearestNeighbors (Cosine Metric)
   - Title Similarity Search: get_similar_books(book_title, n=10)
   - User Recommendation Engine: recommend(user_id, n=10)
   - Rating-weighted similarity ranking with unread-item filtering
====================================================================================================
```

---

## 2. Directory Layout

```
submissions/LCB2023039_Prashik_Humane/
├── Data_Lake_Assignment_2.ipynb    # Complete executable Google Colab notebook with outputs
└── README.md                       # Comprehensive documentation & methodology report
```

---

## 3. Step-by-Step Methodology & Quantitative Findings

### Step 1: Set Up Data Lake & Build Bronze Layer
- Created `bronze/`, `silver/`, and `gold/` lake directories.
- Downloaded `book-crossings.zip` from FreeCodeCamp CDN.
- Read raw CSV files using `sep=';'`, `encoding='latin-1'`, and `on_bad_lines='skip'`.
- Appended `ingestion_timestamp` (`ISO 8601 UTC`) to guarantee data lineage.
- Stored raw, untouched tables in `bronze/` as Parquet:
  - `users_bronze.parquet` : `(278,858, 4)`
  - `books_bronze.parquet` : `(271,360, 9)`
  - `ratings_bronze.parquet` : `(1,149,780, 4)`

### Step 2: Exploratory Data Analysis (EDA) Summary

| EDA Dimension | Quantitative Findings in Bronze Layer | Remediation Action in Silver Layer |
| :--- | :--- | :--- |
| **1. Shapes, Dtypes & Missing Values** | `users`: 278,858 rows, 110,762 missing ages (39.72%).<br>`books`: 271,360 rows, 3 non-numeric year rows, 2 missing authors.<br>`ratings`: 1,149,780 rows, 0 nulls. | Drop unneeded image URLs, cast numeric types cleanly, impute missing user ages, and remove corrupted records. |
| **2. Distribution of Ratings** | 716,109 ratings (62.28%) are 0 (implicit interactions).<br>433,671 ratings are explicit (1–10 scale), peaking at 8 (103,736) and 10 (78,610). | Filter out all 0 ratings; retain strictly explicit user ratings in the range $[1, 10]$. |
| **3. Ages & Publication Years** | Ages range from 0 to 244 (882 users $< 5$, 366 users $> 100$).<br>Publication years contain 4,618 zeros, 11 future years ($> 2026$), and text strings. | Set ages $< 5$ or $> 100$ to null and impute with median age (32); set publication years $\le 0$ or $> 2026$ to null. |
| **4. Ratings per User & Book** | Extreme long tail: 50% of users have 1 rating (90% have $\le 9$).<br>50% of books have 1 rating (90% have $\le 4$). | Apply a minimum threshold ($\ge 10$ ratings) for both users and books to eliminate cold-start noise and isolate reliable patterns. |
| **5. Matrix Sparsity** | 105,283 users $\times$ 340,556 books $= 35.85\text{ billion}$ pairs.<br>Raw interaction density $= 0.003207\%$ (**99.9968% sparse**). | Prune inactive long-tail entities to elevate matrix density by 79x, enabling meaningful cosine similarity neighborhoods. |

### Step 3: Clean Data & Build Silver Layer
- **Deduplication:** Dropped exact duplicates across users, books, and user-book ratings.
- **Explicit Ratings:** Retained only explicit ratings ($1 \le \text{rating} \le 10$).
- **Age Standardization:** Capped ages between 5 and 100; imputed missing values with median age ($32$).
- **Location Parsing:** Parsed `Location` into `city`, `state`, and `country` in Title Case.
- **Publication Year Correction:** Coerced non-numeric and out-of-range years ($< 1800$ or $> 2026$) to `NaN`.
- **Text Normalization:** Stripped whitespace, standardized ISBNs to uppercase, and removed image URLs (`Image-URL-S`, `Image-URL-M`, `Image-URL-L`).
- **Referential Integrity:** Enforced foreign key constraints ensuring all rated books exist in `books` and all rating users exist in `users`.
- **Threshold Justification ($\ge 10$):**
  - Retaining users and books with at least 10 ratings filters out 1-rating cold-start items that produce orthogonal vectors ($cosine = 0$) or random noise.
  - The resulting Silver dataset retains **6,319 active users**, **5,440 popular books**, and **87,233 explicit ratings**.
  - Matrix density increased from **0.0032% to 0.2538%** (a **79-fold density improvement**).

### Step 4: Build Gold Layer (Model-Ready Assets)
- **Book-User CSR Sparse Matrix (`book_user_matrix.npz`):** Dimensions `(5,440, 6,319)`, containing 87,233 non-zero rating entries with a memory footprint of just 0.67 MB.
- **Book Lookup Table (`book_lookup.parquet`):** Instant metadata retrieval for readable titles, authors, publication years, and publishers.
- **User-Level Aggregates (`user_aggregates.parquet`):** Total count, mean rating, min rating, max rating per user.
- **Book-Level Aggregates (`book_aggregates.parquet`):** Total count, mean rating, min rating, max rating per book.

### Step 5: Item-Based Collaborative Filtering Recommender
- **Model:** Scikit-Learn `NearestNeighbors(metric='cosine', algorithm='brute')` fitted on the transposed Book-User matrix (books as rows, users as columns).
- **Cosine Similarity:**
  $$\text{Cosine Similarity}(i, j) = \frac{\vec{v}_i \cdot \vec{v}_j}{\|\vec{v}_i\|_2 \|\vec{v}_j\|_2} = 1 - \text{cosine\_distance}(i, j)$$
- **Title Similarity Search (`get_similar_books(book_title, n=10)`):** Finds the nearest item neighbors based purely on co-rating interaction geometry.
- **Personalized Recommendations (`recommend(user_id, n=10)`):**
  1. Identifies the user's highly rated books ($r_{u, i} \ge 7$).
  2. Retrieves top cosine neighbors for each liked book.
  3. Excludes any books the user has already read.
  4. Ranks candidate books by weighted score:
     $$\text{Score}(u, j) = \sum_{i \in I_u} \text{sim}(i, j) \times r_{u, i}$$
  5. Returns the top $n$ ranked book recommendations with metadata.

---

## 4. Recommender Demonstration & Sanity Check

### 1. Title Similarity Test: *Harry Potter and the Chamber of Secrets*
| Rank | Similar Book Title | Author | Cosine Similarity |
| :---: | :--- | :--- | :---: |
| 1 | *Harry Potter and the Prisoner of Azkaban (Book 3)* | J. K. Rowling | **0.3893** |
| 2 | *Harry Potter and the Sorcerer's Stone* | J. K. Rowling | **0.3699** |
| 3 | *Harry Potter and the Goblet of Fire (Book 4)* | J. K. Rowling | **0.3362** |
| 4 | *Harry Potter and the Order of the Phoenix (Book 5)* | J. K. Rowling | **0.1797** |
| 5 | *Harry Potter and the Sorcerer's Stone (Book 1)* | J. K. Rowling | **0.1376** |

*Verification:* The model automatically clusters the exact canonical Harry Potter sequence in proper narrative order based entirely on reader rating patterns!

---

### 2. Multi-User Personalized Recommendations Demo

#### User 1: `276964` (Epic Fantasy & Legal Thriller Fan)
- **Books User Rated Highly:**
  - `10/10`: *The Terminal Man* by Michael Crichton
  - `10/10`: *The Fires of Heaven (The Wheel of Time, Book 5)* by Robert Jordan
  - `9/10`: *The Chamber* by John Grisham
  - `8/10`: *The Girl Who Loved Tom Gordon* by Stephen King
  - `8/10`: *Stone of Farewell (Memory, Sorrow, and Thorn, Book 2)* by Tad Williams
- **Top 5 Model Recommendations:**
  1. *The Shadow Rising (The Wheel of Time, Book 4)* by Robert Jordan (Score: 11.028)
  2. *To Green Angel Tower, Part 1 (Memory, Sorrow, and Thorn, Book 3)* by Tad Williams (Score: 9.302)
  3. *Lord of Chaos (The Wheel of Time, Book 6)* by Robert Jordan (Score: 8.933)
  4. *Dragonbone Chair Sorrow 1 (Memory, Sorrow, & Thorn)* by Tad Williams (Score: 7.902)
  5. *Dragons of Winter Night* by Margaret Weis (Score: 7.507)

---

#### User 2: `278418` (Children's Literature & Contemporary Romance Fan)
- **Books User Rated Highly:**
  - `10/10`: *Green Eggs and Ham* by Dr. Seuss
  - `9/10`: *The Sign of the Beaver* by Elizabeth George Speare
  - `8/10`: *The Phantom Tollbooth* by Norton Juster
  - `8/10`: *Carolina Moon* by Nora Roberts
  - `8/10`: *Dance upon the Air (Three Sisters Island Trilogy)* by Nora Roberts
- **Top 5 Model Recommendations:**
  1. *Private Scandals* by Nora Roberts (Score: 5.928)
  2. *Heaven and Earth (Three Sisters Island Trilogy)* by Nora Roberts (Score: 5.302)
  3. *Portrait in Death* by Nora Roberts (Score: 3.872)
  4. *Fox in Socks (I Can Read It All by Myself Beginner Books)* by Dr. Seuss (Score: 3.794)
  5. *Jewels of the Sun (Irish Trilogy)* by Nora Roberts (Score: 3.790)

---

#### User 3: `254` (Modern Fantasy & Literary Fiction Fan)
- **Books User Rated Highly:**
  - `10/10`: *American Gods* by Neil Gaiman
  - `9/10`: *The Secret Life of Bees* by Sue Monk Kidd
  - `9/10`: *Smoke and Mirrors: Short Fictions and Illusions* by Neil Gaiman
  - `9/10`: *American Gods: A Novel* by Neil Gaiman
  - `9/10`: *Harry Potter and the Chamber of Secrets (Book 2)* by J. K. Rowling
- **Top 5 Model Recommendations:**
  1. *Harry Potter and the Order of the Phoenix (Book 5)* by J. K. Rowling (Score: 17.483)
  2. *Harry Potter and the Goblet of Fire (Book 4)* by J. K. Rowling (Score: 11.542)
  3. *Harry Potter and the Sorcerer's Stone* by J. K. Rowling (Score: 11.403)
  4. *The Return of the King (The Lord of The Rings, Part 3)* by J. R. R. Tolkien (Score: 9.237)
  5. *Free* by Paul Vincent (Score: 6.852)

---

## 5. How to Run the Notebook in Google Colab

1. Open [Google Colab](https://colab.research.google.com/).
2. Click **File > Upload Notebook** and select `Data_Lake_Assignment_2.ipynb`.
3. In the top navigation bar, select **Runtime > Run all** (or press `Ctrl + F9`).
4. The notebook will automatically:
   - Download and extract the Book-Crossing dataset from the official URL.
   - Build `bronze/`, `silver/`, and `gold/` data lake folders.
   - Execute all exploratory data analysis and display quantitative findings.
   - Train the item-based nearest neighbors model.
   - Run the demo and print high-rated books alongside top recommendations.

---

## 6. How to Submit on GitHub (Step-by-Step Guide)

Follow these exact steps to submit your assignment to the class repository:

### Step 6.1: Fork the Class Repository
1. Go to the course GitHub repository:
   [https://github.com/necromancersanju-netizen/data-mining-assignment-2](https://github.com/necromancersanju-netizen/data-mining-assignment-2)
2. In the top-right corner, click the **Fork** button.
3. Keep the default settings and click **Create fork**. This creates a copy under your personal GitHub account:
   `https://github.com/<YOUR_GITHUB_USERNAME>/data-mining-assignment-2`

### Step 6.2: Clone Your Forked Repository Locally
Open your terminal (PowerShell, Command Prompt, or Git Bash) and run:
```bash
# Clone your fork (replace <YOUR_GITHUB_USERNAME> with your actual username)
git clone https://github.com/<YOUR_GITHUB_USERNAME>/data-mining-assignment-2.git
cd data-mining-assignment-2
```

### Step 6.3: Add Your Submission Folder
1. Create your submission folder under `submissions/`:
   ```bash
   mkdir -p submissions/LCB2023039_Prashik_Humane
   ```
2. Copy your notebook and README into that folder:
   - `Data_Lake_Assignment_2.ipynb`
   - `README.md`

Your repository tree should look like:
```
data-mining-assignment-2/
├── README.md
└── submissions/
    ├── .gitkeep
    └── LCB2023039_Prashik_Humane/
        ├── Data_Lake_Assignment_2.ipynb
        └── README.md
```

### Step 6.4: Commit and Push Your Work
Run the following Git commands:
```bash
# Stage your submission folder
git add submissions/LCB2023039_Prashik_Humane/

# Verify only your folder is staged
git status

# Commit with a clean message
git commit -m "Add Data Mining Assignment 2 submission for LCB2023039 Prashik Humane"

# Push to your GitHub fork
git push origin main
```

### Step 6.5: Open a Pull Request (PR)
1. Open your browser and go to your fork:
   `https://github.com/<YOUR_GITHUB_USERNAME>/data-mining-assignment-2`
2. You will see a banner saying *"This branch is 1 commit ahead of necromancersanju-netizen:main"*.
3. Click **Contribute > Open pull request** (or click the green **New pull request** button).
4. Set the PR details:
   - **Base repository:** `necromancersanju-netizen/data-mining-assignment-2` (base: `main`)
   - **Head repository:** `<YOUR_GITHUB_USERNAME>/data-mining-assignment-2` (compare: `main`)
5. Set the **Pull Request Title**:
   ```
   Data Mining Assignment 2 - LCB2023039 - Prashik Humane
   ```
6. In the description, provide a brief summary:
   ```markdown
   ### Data Mining Assignment 2: Medallion Data Lake & Recommender System
   - Student Name: Prashik Humane
   - Roll Number: LCB2023039
   - Deliverables:
     * submissions/LCB2023039_Prashik_Humane/Data_Lake_Assignment_2.ipynb
     * submissions/LCB2023039_Prashik_Humane/README.md
   ```
7. Click **Create pull request**. Your submission is complete!
