
import pandas as pd
import numpy as np

import streamlit as st

from scipy.sparse import load_npz
from sklearn.neighbors import NearestNeighbors


# --------------------------------------------------
# LOAD GOLD DATA
# --------------------------------------------------

@st.cache_resource
def load_model():

    book_lookup = pd.read_parquet(
        "gold/book_lookup.parquet"
    )

    user_book_matrix = load_npz(
        "gold/user_book_matrix.npz"
    )

    user_ids = pd.read_pickle(
        "gold/user_ids.pkl"
    ).to_numpy()

    book_ids = pd.read_pickle(
        "gold/book_ids.pkl"
    ).to_numpy()

    # Normalize ISBNs for consistent matching
    book_ids = book_ids.astype(str)
    book_lookup["ISBN"] = book_lookup["ISBN"].astype(str)

    # Create index mappings
    user_to_index = {
        int(user_id): index
        for index, user_id in enumerate(user_ids)
    }

    book_to_index = {
        isbn: index
        for index, isbn in enumerate(book_ids)
    }

    # Convert user-book matrix into book-user matrix
    book_user_matrix = user_book_matrix.T.tocsr()

    # Train KNN model
    knn_model = NearestNeighbors(
        metric="cosine",
        algorithm="brute",
        n_neighbors=11,
        n_jobs=-1
    )

    knn_model.fit(book_user_matrix)

    return (
        book_lookup,
        user_book_matrix,
        book_user_matrix,
        user_ids,
        book_ids,
        user_to_index,
        book_to_index,
        knn_model
    )


(
    book_lookup,
    user_book_matrix,
    book_user_matrix,
    user_ids,
    book_ids,
    user_to_index,
    book_to_index,
    knn_model
) = load_model()


# --------------------------------------------------
# SIMILAR BOOK RECOMMENDATIONS
# --------------------------------------------------

def get_similar_books(book_title, n=10):

    matched_books = book_lookup[
        book_lookup["Book-Title"].str.contains(
            book_title,
            case=False,
            na=False,
            regex=False
        )
    ]

    if matched_books.empty:
        return pd.DataFrame(
            columns=[
                "ISBN",
                "Book-Title",
                "Book-Author",
                "Similarity"
            ]
        )

    selected_book = matched_books.iloc[0]

    isbn = selected_book["ISBN"]

    book_index = book_to_index.get(isbn)

    if book_index is None:
        return pd.DataFrame(
            columns=[
                "ISBN",
                "Book-Title",
                "Book-Author",
                "Similarity"
            ]
        )

    distances, indices = knn_model.kneighbors(
        book_user_matrix[book_index],
        n_neighbors=min(n + 1, len(book_ids))
    )

    similar_indices = indices[0][1:]
    similar_distances = distances[0][1:]

    similarity_scores = 1 - similar_distances

    similar_isbns = book_ids[similar_indices]

    results = book_lookup[
        book_lookup["ISBN"].isin(similar_isbns)
    ].copy()

    score_mapping = dict(
        zip(similar_isbns, similarity_scores)
    )

    results["Similarity"] = results["ISBN"].map(
        score_mapping
    )

    results = results.sort_values(
        "Similarity",
        ascending=False
    ).head(n)

    return results[
        ["ISBN", "Book-Title", "Book-Author", "Similarity"]
    ].reset_index(drop=True)


# --------------------------------------------------
# PERSONALIZED RECOMMENDATIONS
# --------------------------------------------------

def recommend(user_id, n=10):

    empty_result = pd.DataFrame(
        columns=[
            "ISBN",
            "Book-Title",
            "Book-Author",
            "Recommendation-Score"
        ]
    )

    user_index = user_to_index.get(int(user_id))

    if user_index is None:
        return empty_result

    # Get user's ratings directly from sparse matrix
    user_row = user_book_matrix.getrow(user_index)

    rated_indices = user_row.indices
    ratings = user_row.data

    if len(rated_indices) == 0:
        return empty_result

    # Select highly rated books
    liked_books = [
        (book_index, rating)
        for book_index, rating in zip(
            rated_indices,
            ratings
        )
        if rating >= 7
    ]

    if not liked_books:
        return empty_result

    already_rated = set(rated_indices)

    recommendation_scores = {}

    # Find similar books for liked books
    for book_index, rating in liked_books:

        distances, indices = knn_model.kneighbors(
            book_user_matrix[book_index],
            n_neighbors=min(11, len(book_ids))
        )

        for similar_index, distance in zip(
            indices[0][1:],
            distances[0][1:]
        ):

            if similar_index in already_rated:
                continue

            similarity = 1 - distance

            score = similarity * rating

            recommendation_scores[similar_index] = (
                recommendation_scores.get(similar_index, 0)
                + score
            )

    if not recommendation_scores:
        return empty_result

    results = pd.DataFrame(
        [
            (book_ids[index], score)
            for index, score in recommendation_scores.items()
        ],
        columns=["ISBN", "Recommendation-Score"]
    )

    results = results.merge(
        book_lookup,
        on="ISBN",
        how="left"
    )

    results = results.sort_values(
        "Recommendation-Score",
        ascending=False
    ).head(n)

    return results[
        [
            "ISBN",
            "Book-Title",
            "Book-Author",
            "Recommendation-Score"
        ]
    ].reset_index(drop=True)
