# LCB2023035 — Lavish Shakya — Data Mining Assignment 2

Book recommender built on a medallion data lake (Bronze → Silver → Gold) over the
Book-Crossing dataset, using item-based collaborative filtering with cosine similarity.

## Contents

| Path | What it holds |
|---|---|
| `LCB2023035_Lavish_Shakya.ipynb` | the full pipeline with all outputs and figures |
| `bronze/` | the three raw CSVs copied to Parquet, untouched, every column still a string |
| `silver/` | typed, de-duplicated, validated data, filtered to users and books with ≥ 10 ratings |
| `gold/` | sparse rating matrix (`.npz`), user and book index tables, book lookup, aggregates |

## Pipeline summary

| Stage | Ratings | Users | Books |
|---|---|---|---|
| Bronze (raw) | 1,149,780 | 278,858 | 271,379 |
| after dropping implicit 0 ratings | 433,671 | — | — |
| after referential integrity | 383,853 | — | — |
| Silver (≥ 10 ratings each, iterated to a fixed point) | 41,469 | 1,820 | 2,030 |

Density rises from 0.001% to 1.12%. The Gold matrix is 0.09 MB sparse against 14 MB dense.

## Results

Leave-one-out test over 400 sampled users — one liked book hidden, recommendations
generated from the rest, a hit if it returns in the top 10:

| Model | Hit rate @10 |
|---|---|
| item-based collaborative filtering | **36.2%** |
| most-rated-books baseline | 5.5% |

6.6× the popularity baseline, with a median recovered rank of 2 of 10.

## Running it

Open the notebook in Colab and choose **Runtime → Run all**, or run it locally with
`pandas`, `pyarrow`, `numpy`, `scipy`, `scikit-learn` and `matplotlib` installed. It
downloads the dataset itself and rebuilds all three layers from scratch, so the
committed Parquet files are reproducible output rather than inputs.
