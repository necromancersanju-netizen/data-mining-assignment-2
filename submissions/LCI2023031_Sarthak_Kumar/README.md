
# Book Recommender System

**Data Mining Assignment 2**

- **Name:** Sarthak Kumar
- **Roll Number:** LCI2023031

## Project Overview

This project implements a Book Recommendation System using the Book-Crossing dataset. It follows the Bronze-Silver-Gold Medallion Data Lake Architecture to ingest, clean, transform, and prepare data for recommendations.

The recommendation engine uses Item-Based Collaborative Filtering with cosine similarity to suggest books based on a user's highly rated books.

## Dataset

- **Dataset:** Book-Crossing Dataset
- **Source:** https://cdn.freecodecamp.org/project-data/books/book-crossings.zip
- **Files:** Users, Books, and Ratings

## Technologies Used

- Python
- Google Colab
- Pandas and NumPy
- SciPy
- Scikit-learn
- Matplotlib and Seaborn
- Parquet

## Data Lake Architecture

### 1. Bronze Layer

- Downloaded the original Users, Books, and Ratings datasets.
- Loaded CSV files using the appropriate separator and encoding.
- Added an ingestion timestamp to each dataset.
- Stored the datasets in Parquet format without applying cleaning operations.

### 2. Silver Layer

Performed data cleaning and preprocessing:

- Removed duplicate records.
- Removed zero ratings and retained ratings from 1 to 10.
- Handled invalid and missing user ages using median imputation.
- Split the Location column into City, State, and Country.
- Handled invalid publication years.
- Standardized text columns and removed image URL columns.
- Maintained referential integrity between users, books, and ratings.
- Applied a minimum threshold of 10 ratings per user and per book.

### 3. Gold Layer

Created analytical datasets for recommendation:

- Sparse user-book rating matrix.
- Book lookup table containing book titles and authors.
- User-level aggregates.
- Book-level aggregates.

## Recommendation System

Implemented Item-Based Collaborative Filtering using:

- Cosine similarity.
- Scikit-learn's `NearestNeighbors` algorithm.
- Sparse user-book rating matrix.

The recommendation function:

- Identifies books highly rated by a user.
- Finds similar books using cosine similarity.
- Calculates recommendation scores weighted by the user's ratings.
- Excludes books the user has already rated.
- Returns the top 10 recommended books.

## Testing and Validation

The recommendation system was tested on three different users.

Validation confirmed that:

- Each tested user received 10 recommendations.
- Previously rated books were excluded.
- Recommendations were sorted in descending order of recommendation score.
- The Silver layer contained no duplicate user-book pairs.
- All retained users and books satisfied the minimum rating threshold of 10.

## How to Run

1. Open the `Book_Recommender_System.ipynb` notebook in Google Colab.
2. Run the notebook cells sequentially from top to bottom.
3. The notebook downloads the Book-Crossing dataset and creates the Bronze, Silver, and Gold layers.
4. Execute the recommendation section to test recommendations for sample users.

## Conclusion

The project demonstrates an end-to-end Book Recommendation System using Medallion Data Lake Architecture and Item-Based Collaborative Filtering. It covers data ingestion, exploratory data analysis, preprocessing, sparse matrix construction, similarity computation, and personalized book recommendations.
