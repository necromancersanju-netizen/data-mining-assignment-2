"""Test the notebook's real definitions without downloading its dataset."""
import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

SOURCE = Path(__file__).parents[1] / "book_recommender.py"


@pytest.fixture
def definitions():
    tree = ast.parse(SOURCE.read_text())
    nodes = [
        node for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom, ast.FunctionDef, ast.ClassDef))
    ]
    namespace = {}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec"), namespace)
    return namespace


def test_iterative_core_removes_cascading_low_degree_nodes(definitions):
    ratings = pd.DataFrame({
        "user_id": [1, 1, 2, 2, 3, 3, 4],
        "isbn": ["a", "b", "a", "b", "b", "c", "c"],
        "rating": [8] * 7,
    })
    result, history = definitions["filter_core"](ratings, 2)
    assert set(result.user_id) == {1, 2}
    assert set(result.isbn) == {"a", "b"}
    assert len(history) > 1
    assert result.groupby("user_id").size().min() >= 2
    assert result.groupby("isbn").size().min() >= 2


def test_clean_ages_preserves_boundaries_and_imputes_valid_median(definitions):
    ages, median = definitions["clean_ages"](
        pd.Series(["4", "5", "25", "100", "101", "NULL", "bad"])
    )
    assert median == 25
    assert ages.tolist() == [25, 5, 25, 100, 25, 25, 25]


def test_all_invalid_ages_cannot_invent_median(definitions):
    with pytest.raises(ValueError, match="valid ages"):
        definitions["clean_ages"](pd.Series(["NULL", "200"]))


def test_core_reports_empty_instead_of_lowering_threshold(definitions):
    ratings = pd.DataFrame({"user_id": [1], "isbn": ["a"], "rating": [8]})
    result, history = definitions["filter_core"](ratings, 2)
    assert result.empty
    assert history[-1]["users"] == history[-1]["books"] == 0


def test_text_normalization_preserves_display_case_and_normalizes_matching(definitions):
    assert definitions["normalize_text"](pd.Series(["  A   Title\n"])).iloc[0] == "A Title"
    assert definitions["title_key"]("  A   TITLE\n") == "a title"


@pytest.fixture
def model(definitions):
    # Rows are books, columns users. User 11 has read a and d.
    matrix = csr_matrix(np.array([
        [10, 8, 0], [0, 8, 0], [0, 0, 9], [8, 0, 9], [0, 7, 2],
    ], dtype=np.float32))
    lookup = pd.DataFrame({
        "book_idx": range(5), "isbn": list("abcde"),
        "title": ["Alpha", "Beta", "Gamma", "Delta", "Epsilon"],
        "author": ["Author"] * 5,
    })
    users = pd.DataFrame({"user_idx": range(3), "user_id": [11, 22, 33]})
    return definitions["BookRecommender"](matrix, lookup, users, neighbor_count=4)


def test_similar_books_excludes_self_and_zero_similarity(model):
    result = model.similar_books("  ALPHA  ", n=10)
    assert "a" not in set(result.isbn)
    assert "c" not in set(result.isbn)
    assert result.similarity.between(0, 1, inclusive="right").all()
    assert result.similarity.is_monotonic_decreasing


def test_recommend_excludes_read_books_and_uses_rating_weighted_similarity(model):
    result = model.recommend(11, n=10)
    assert set(result.isbn) == {"b", "c", "e"}
    assert result.isbn.is_unique
    assert result.score.is_monotonic_decreasing
    matrix = model.matrix.toarray()
    expected = {}
    for candidate in [1, 2, 4]:
        expected[model.lookup.iloc[candidate].isbn] = sum(
            np.dot(matrix[candidate], matrix[seed])
            / (np.linalg.norm(matrix[candidate]) * np.linalg.norm(matrix[seed]))
            * rating / 10
            for seed, rating in [(0, 10), (3, 8)]
        )
    for row in result.itertuples():
        assert row.score == pytest.approx(expected[row.isbn], abs=1e-6)


def test_unknown_user_and_title_are_explicit_errors(model):
    with pytest.raises(ValueError, match="Unknown user"):
        model.recommend(999)
    with pytest.raises(ValueError, match="Unknown title"):
        model.similar_books("not a book")


def test_no_high_ratings_returns_empty_without_popularity_fallback(model):
    assert model.recommend(22, high_rating=10).empty
    assert model.recommend(11, n=0).empty
    with pytest.raises(ValueError, match="n must"):
        model.recommend(11, n=-1)


def test_duplicate_title_requires_isbn(model, definitions):
    lookup = model.lookup.copy()
    lookup.loc[1, "title"] = "Alpha"
    model = definitions["BookRecommender"](model.matrix, lookup, model.users)
    with pytest.raises(ValueError, match="Ambiguous"):
        model.similar_books("Alpha")
    result = model.similar_books("Alpha", isbn="a")
    assert "a" not in set(result.isbn)


def test_implicit_interactions_and_other_editions_are_excluded(model, definitions):
    known = pd.DataFrame({
        "user_id": [11, 11], "isbn": ["b", "other-edition-of-c"],
        "title_key": ["beta", "gamma"], "author_key": ["author", "author"],
    })
    model = definitions["BookRecommender"](
        model.matrix, model.lookup, model.users, known=known, neighbor_count=4
    )
    assert model.recommend(11).isbn.tolist() == ["e"]


def test_mismatched_gold_mapping_is_rejected(model, definitions):
    with pytest.raises(ValueError, match="shapes"):
        definitions["BookRecommender"](model.matrix, model.lookup.iloc[:-1], model.users)
