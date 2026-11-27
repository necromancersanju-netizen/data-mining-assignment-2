# 📚 Book Recommendation System

An intelligent Book Recommendation System built using **Python, Collaborative Filtering, Scikit-learn, and Streamlit**.

The project uses the Book-Crossing dataset to recommend books based on user rating patterns and similarities between books.

## 🚀 Project Overview

This project implements an Item-Based Collaborative Filtering approach to discover books that readers may enjoy.

It provides two recommendation features:

- **Similar Books Recommendation:** Find books similar to a selected book.
- **Personalized Book Recommendation:** Recommend books based on a user's previous ratings.

## ✨ Features

- Book similarity recommendations using Cosine Similarity.
- Personalized recommendations based on highly rated books.
- Sparse User-Book Rating Matrix.
- K-Nearest Neighbors (KNN) model.
- Data preprocessing using Medallion Architecture.
- Interactive Streamlit web application.

## 🏗️ Project Architecture

The project follows the Medallion Data Lake Architecture.

| Layer | Description |
|---|---|
| Bronze | Stores raw Users, Books, and Ratings data |
| Silver | Data cleaning, transformation, and filtering |
| Gold | Processed data, aggregates, and sparse rating matrix |

## 🧠 Recommendation Methodology

### 1. Item-Based Collaborative Filtering

The system identifies similarities between books using users' historical ratings.

**Workflow:**

1. Create a User-Book Rating Matrix.
2. Convert it into a Book-User Matrix.
3. Calculate book similarities using Cosine Distance.
4. Use K-Nearest Neighbors to identify similar books.
5. Return the top 10 similar books.

### 2. Personalized Recommendations

The system uses a user's highly rated books to generate personalized suggestions.

**Workflow:**

1. Retrieve the user's rating history.
2. Identify books rated 7 or higher.
3. Find similar books for each highly rated book.
4. Calculate recommendation scores using similarity and user ratings.
5. Exclude books the user has already rated.
6. Return the top recommended books.

## 🛠️ Tech Stack

- **Programming:** Python
- **Data Processing:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Sparse Matrix:** SciPy
- **Visualization:** Matplotlib, Seaborn
- **Web Application:** Streamlit
- **Storage:** Parquet, Pickle, NPZ
- **Version Control:** Git and GitHub

## 📂 Project Structure

```text
book-recommender-system/
│
├── app.py
├── recommender.py
├── requirements.txt
├── README.md
│
└── gold/
    ├── book_lookup.parquet
    ├── book_aggregates.parquet
    ├── user_aggregates.parquet
    ├── user_book_matrix.npz
    ├── user_ids.pkl
    └── book_ids.pkl
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/book-recommender-system.git
```

### 2. Navigate to the project directory

```bash
cd book-recommender-system
```

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Run the application

```bash
python -m streamlit run app.py
```

The application will open in your browser.

## 🌐 Deployment

The application is designed for deployment using **Streamlit Community Cloud**.

Deployment steps:

1. Upload the project to GitHub.
2. Open Streamlit Community Cloud.
3. Connect your GitHub repository.
4. Select `app.py` as the main application file.
5. Deploy the application.

## 🎯 Project Objective

The main objective is to build an end-to-end book recommendation system that combines data engineering concepts, collaborative filtering, machine learning, and interactive application development.

---

**Developed using Python and Machine Learning.**
