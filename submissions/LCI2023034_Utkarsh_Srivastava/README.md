# LCI2023034 - Utkarsh Srivastava - Data Mining Assignment 2

Book recommender built on a medallion data lake (Bronze → Silver → Gold) using the Book-Crossing dataset.

- `LCI2023034_Utkarsh_Srivastava.ipynb`: the full pipeline, with outputs
- `bronze/`: raw books, ratings and users stored as Parquet
- `silver/`: cleaned data (books and users with at least 10 ratings each)
- `gold/`: user–book rating matrix (`.npz`), index and lookup tables, and user and book statistics

To run in Colab: **Runtime > Run all**. The notebook downloads the dataset itself and rebuilds all three layers.
