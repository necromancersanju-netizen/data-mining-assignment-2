# Book Recommender System Using Medallion Data Lake

## Overview

This project implements a book recommendation system using the Book-Crossing dataset. It follows the Medallion Data Lake architecture, organizing data into Bronze, Silver, and Gold layers before building an item-based collaborative filtering model.

The system uses cosine similarity between books to recommend books based on users' previous ratings.

## Dataset

The project uses the Book-Crossing dataset, containing:
- Users
- Books
- Book ratings

Dataset source: [Book-Crossing Dataset](https://cdn.freecodecamp.org/project-data/books/book-crossings.zip)

## Project Architecture

### 1. Bronze Layer
- Downloaded and ingested the original Users, Books, and Ratings datasets.
- Preserved the raw data.
- Added ingestion timestamps.
- Stored the datasets in Parquet format.

### 2. Silver Layer
- Removed duplicate records.
- Removed zero ratings and retained explicit ratings from 1 to 10.
- Handled invalid user ages and publication years.
- Split location information into city, state, and country.
- Cleaned text fields and removed image URL columns.
- Retained users and books with at least 10 ratings.

### 3. Gold Layer
- Created a sparse user-book rating matrix.
- Generated user and book mappings.
- Created a book lookup table.
- Calculated user-level and book-level rating aggregates.

### 4. Recommendation Model
- Implemented item-based collaborative filtering.
- Used cosine similarity to identify similar books.
- Weighted recommendations using users' highly rated books.
- Excluded books already rated by the user.
- Generated the top 10 recommendations for each user.

## Technologies Used

- Python
- Pandas
- NumPy
- SciPy
- Scikit-learn
- Matplotlib
- Seaborn
- PyArrow
- Google Colab

## Results

After preprocessing and filtering:

| Dataset | Records |
|---|---:|
| Users | 1,813 |
| Books | 2,022 |
| Ratings | 41,238 |

The final sparse user-book matrix has 1,813 rows and 2,022 columns, with 41,238 stored ratings.

The recommendation model was tested on three different users and successfully generated personalized top-10 book recommendations.

## How to Run

1. Open `Book_Recommender_System.ipynb` in Google Colab.
2. Run the notebook cells sequentially.
3. The notebook downloads the dataset and creates the Bronze, Silver, and Gold layers.
4. The item-based collaborative filtering model generates similar-book results and personalized recommendations.

## Future Improvements

- Incorporate book genres and descriptions using content-based filtering.
- Experiment with different similarity metrics.
- Evaluate recommendations using precision@k, recall@k, and other ranking metrics.
- Handle multiple ISBN editions of the same book to reduce duplicate recommendations.
