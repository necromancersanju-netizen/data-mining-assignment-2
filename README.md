# Book Recommender with a Medallion Data Lake

Item-based collaborative filtering recommender on the **Book-Crossing** dataset, structured as a **Bronze → Silver → Gold** medallion data lake in [submissions/recommender_medallion.ipynb](submissions/recommender_medallion.ipynb).

## Architecture

1. **Bronze (`bronze/*.parquet`)**: Downloads `book-crossings.zip`, parses `BX-Users.csv`, `BX-Books.csv`, and `BX-Book-Ratings.csv`, and stores raw tables with an `ingestion_timestamp`.
2. **EDA**: Inspects table shapes, missing values, implicit (`0`) vs. explicit (`1–10`) rating distributions, age/year anomalies, long-tail user/book rating counts, and matrix sparsity (`>99.99%`).
3. **Silver (`silver/*.parquet`)**:
   - Filters to explicit ratings (`1–10`) and deduplicates `(User-ID, ISBN)` pairs.
   - Cleans `Age` (nulls outside `5–100`, median-imputed), parses `Location` into `city`/`state`/`country`, normalizes `Year-Of-Publication` (`0` and future years set to `NaN`), and standardizes book metadata.
   - Iteratively prunes users and books with fewer than `10` ratings, yielding `1,820` users, `2,030` books, and `41,456` ratings.
4. **Gold (`gold/*`)**: Builds the sparse CSR `user_book_matrix.npz` (`1820 × 2030`, `1.1221%` density) along with `user_index.parquet`, `book_index.parquet`, `book_lookup.parquet`, `user_stats.parquet`, and `book_stats.parquet`.
5. **Model & Demo**: Fits `NearestNeighbors(metric='cosine', algorithm='brute')` on the transposed item-user matrix to power `similar_books(title, n=10)` and `recommend(user_id, n=10)`.
6. **Offline Evaluation**: Evaluates top-10 recommendations on an 80/20 per-user train/test split (`1,766` users with `>= 1` held-out book rated `>= 7`):

| Model | Precision@10 | Recall@10 | HitRate@10 | NDCG@10 | CatalogCoverage@10 |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Item-Based CF (Cosine, k=50)** | **0.0404** | **0.1143** | **0.2888** | **0.0974** | **0.7985** |
| Popularity Baseline | 0.0152 | 0.0409 | 0.1365 | 0.0322 | 0.0094 |

## How to Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace submissions/recommender_medallion.ipynb
```
