# Book Recommendation System

## Overview
Implemented a book recommendation system using the Book-Crossing dataset and a Medallion Data Lake architecture.

## Technologies Used
- Python
- Pandas and NumPy
- SciPy
- Scikit-learn
- Google Colab
- Parquet

## Project Structure
- **Bronze:** Raw dataset ingestion.
- **Silver:** Data cleaning and preprocessing.
- **Gold:** Sparse user-book rating matrix, lookup tables and aggregates.
- **Recommendation Engine:** Item-based collaborative filtering using cosine similarity.

## Features
- Exploratory Data Analysis (EDA)
- Data cleaning and validation
- Similar book recommendations
- Personalized user-based book recommendations
- Top 10 recommendations
- Recommendation evaluation using Precision@10, Recall@10 and Hit Rate@10.

## Execution
Open `Recommender_System_Assignment.ipynb` in Google Colab and execute the cells sequentially.

## Dataset
[Book-Crossing Dataset](https://cdn.freecodecamp.org/project-data/books/book-crossings.zip)