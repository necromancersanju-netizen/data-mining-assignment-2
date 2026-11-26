# Book-Crossing Medallion Recommender

A Google Colab-ready solution to the Data Mining Lab assignment: raw CSVs →
Bronze Parquet → cleaned Silver → sparse Gold → item-based book recommendations.

## Run in Google Colab

1. Open [Google Colab](https://colab.research.google.com/).
2. Choose **File → Upload notebook** and upload `book_recommender.ipynb`.
3. Use a standard **CPU** runtime, then select **Runtime → Run all**.

The notebook is self-contained: it installs its dependencies, downloads the
provided dataset, creates `data_lake/bronze`, `silver`, and `gold`, and runs the
demos. No companion Python files, credentials, GPU, or Google Drive are needed.
The checked-in notebook includes real executed outputs; rerunning recomputes them.
Execution takes a few minutes, depending on the runtime and network.

## Files

| File | Purpose |
|---|---|
| `book_recommender.ipynb` | **Main submission**, including EDA, explanations, plots and demos |
| `book_recommender.py` | Matching percent-cell source, for readable review and maintenance |
| `build_notebook.py` | Rebuilds the standalone notebook, optionally executing every cell |
| `requirements.txt` | Local development/test dependencies |
| `tests/test_recommender.py` | Small synthetic tests for cleaning, filtering and recommendation behavior |

Generated data stays out of Git. The original ZIP and CSVs are retained under
`data_lake/raw/`; a source SHA-256 and ingestion timestamp are saved in Bronze.

## Method and design decisions

- **Bronze:** semicolon-separated, Latin-1 CSVs with backslash escaping; validate
  field counts, preserve all source strings, add a UTC ingestion timestamp,
  and verify Parquet round trips. Literal `NULL` is preserved until Silver.
- **EDA:** shapes/types, actual and semantic missingness, rating/age/year
  distributions, interaction counts, and both union-catalog and metadata-matched
  pair density/sparsity. Markdown findings state the Silver action.
- **Silver:** deduplicate, keep explicit ratings 1–10, impute invalid/missing
  ages using the valid median, split locations, normalize text, remove image URLs,
  null invalid years, and enforce user/book foreign keys. Repeated user–book
  ratings are averaged because the source has no rating timestamp.
- **Threshold:** compare iterative 5/10/15/20 cores; use **10 ratings per user
  and per ISBN**, iterating until both constraints hold. This is a justified
  coverage/support heuristic, not a tuned accuracy claim.
- **Gold:** CSR `book × user` matrix, row/column mappings, readable book lookup,
  user/book aggregates, and known-interaction metadata for exclusion.
- **Model:** sparse cosine nearest neighbors. Each seed rated **≥8** contributes
  `(rating / 10) × cosine similarity` to its 50 closest positive-similarity
  neighbors. Sum contributions and return up to ten unseen unique works.
- **Exclusions:** all known interactions, including raw implicit zeros, and
  obvious alternate editions with matching normalized title/author.
- **Demo:** three deterministic users from low/medium/high activity bands,
  with highly rated books next to recommendations.

Unknown users/titles raise descriptive errors; ambiguous titles require an ISBN.
Users without high ratings return an empty table. Sparse candidates may yield
fewer than requested recommendations. This demo is not a held-out evaluation:
no accuracy claims are made. Cold-start coverage and small co-rating overlap
remain limitations.

## Verified full-data run

- Bronze: **278,858 users**, **271,379 books**, **1,149,780 ratings**, with no skipped records.
- Implicit zero ratings: **716,109**; explicit ratings before metadata filtering: **433,671**.
- Final Gold: **2,030 × 1,820**, with **41,469** nonzero ratings (1.1224% density).
- Demo users: **805**, **143964**, **11676**, each with ten recommendations.
- Full notebook execution and its data-integrity/recommendation assertions passed.

## Local reproduction and tests

From this submission directory, using Python 3.12 or later:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pytest tests -q
python build_notebook.py --execute
```

`build_notebook.py --execute` rebuilds from source and executes in a fresh Python
kernel. Omit `--execute` to rebuild without outputs. The notebook uses only its
own code, not imports from the source file. Tests extract imports and definitions
from the same source without running the download or full-data cells.

The default lake location is relative to the notebook. Locally, set
`BOOK_LAKE_ROOT` to use another directory. Delete/move a cached archive if you
deliberately want to redownload it. The year cutoff is the current UTC year,
so invalid-year counts may change in future years.

## GitHub submission

Place this entire folder under the assignment repository's `submissions/`
directory, following any instructor naming convention:

```text
submissions/book_crossing_medallion/
```

If contributing to someone else's repository, use a fork and pull request as
required by the instructor. The notebook and README are the deliverables;
generated datasets and lake artifacts are reproducible and should not be uploaded.

Dataset: <https://cdn.freecodecamp.org/project-data/books/book-crossings.zip>
