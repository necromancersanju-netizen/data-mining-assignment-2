import streamlit as st
import pandas as pd
from recommender import get_similar_books, recommend

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Book Recommendation System",
    page_icon="📚",
    layout="wide"
)

# --------------------------------------------------
# SESSION STATE INITIALIZATION
# --------------------------------------------------
if "reading_list" not in st.session_state:
    st.session_state.reading_list = []

if "search_history" not in st.session_state:
    st.session_state.search_history = []

if "similar_results" not in st.session_state:
    st.session_state.similar_results = None

if "personal_results" not in st.session_state:
    st.session_state.personal_results = None


def add_to_reading_list(book_row: dict):
    """Add a book dictionary to the session reading list if not already saved."""
    if book_row not in st.session_state.reading_list:
        st.session_state.reading_list.append(book_row)
        st.toast(f"Saved '{book_row.get('Title', book_row.get('Book-Title', 'Book'))}' to Reading List!", icon="🔖")
    else:
        st.toast("Already in your Reading List!", icon="ℹ️")


def render_book_cards(df: pd.DataFrame, key_prefix: str):
    """Render recommendations as interactive cards with 'Save' buttons."""
    # Dynamically detect title, author, and score columns
    cols = df.columns.tolist()
    title_col = next((c for c in cols if "title" in c.lower()), cols[0])
    author_col = next((c for c in cols if "author" in c.lower()), None)
    score_col = next((c for c in cols if any(k in c.lower() for k in ["sim", "score", "rating", "match"])), None)

    for idx, row in df.iterrows():
        title = str(row[title_col])
        author = f"by **{row[author_col]}**" if author_col and pd.notna(row[author_col]) else ""
        score_val = row[score_col] if score_col and pd.notna(row[score_col]) else None

        with st.container(border=True):
            c1, c2, c3 = st.columns([6, 2, 2])
            with c1:
                st.markdown(f"### 📖 {title}")
                if author:
                    st.caption(author)
            with c2:
                if score_val is not None:
                    if isinstance(score_val, float) and score_val <= 1.0:
                        st.metric("Match Score", f"{score_val * 100:.1f}%")
                    else:
                        st.metric("Score / Rating", f"{score_val:.2f}" if isinstance(score_val, float) else score_val)
            with c3:
                st.write("")  # Vertical spacing
                if st.button("🔖 Save Book", key=f"{key_prefix}_save_{idx}", use_container_width=True):
                    add_to_reading_list(row.to_dict())


# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------
st.markdown("""
<style>
    .main-title {
        font-size: 38px;
        font-weight: 700;
        color: #4F46E5;
    }
    .subtitle {
        font-size: 17px;
        color: #888888;
    }
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown('<div class="main-title">📚 Book Recommendation System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Discover, filter, and save your next favourite read.</div>', unsafe_allow_html=True)

# --------------------------------------------------
# SIDEBAR: CONTROLS & READING LIST
# --------------------------------------------------
with st.sidebar:
    st.header("⚙️ Controls")
    num_recs = st.slider("Number of recommendations", min_value=3, max_value=25, value=10)
    view_mode = st.radio("Display format", ["Interactive Cards", "Data Table"], horizontal=True)

    st.divider()

    # Recent Search History
    if st.session_state.search_history:
        st.subheader("🕒 Recent Searches")
        for past_query in reversed(st.session_state.search_history[-5:]):
            if st.button(f"🔎 {past_query}", key=f"hist_{past_query}", use_container_width=True):
                st.session_state.prefill_book = past_query

    st.divider()

    # Saved Reading List
    st.subheader(f"🔖 Reading List ({len(st.session_state.reading_list)})")
    if st.session_state.reading_list:
        saved_df = pd.DataFrame(st.session_state.reading_list)
        title_col = next((c for c in saved_df.columns if "title" in c.lower()), saved_df.columns[0])

        for i, item in enumerate(st.session_state.reading_list):
            col_a, col_b = st.columns([4, 1])
            col_a.write(f"• {item[title_col]}")
            if col_b.button("✖", key=f"del_{i}", help="Remove"):
                st.session_state.reading_list.pop(i)
                st.rerun()

        csv_data = saved_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Export Reading List (CSV)",
            data=csv_data,
            file_name="my_reading_list.csv",
            mime="text/csv",
            use_container_width=True
        )
        if st.button("🗑️ Clear List", use_container_width=True):
            st.session_state.reading_list = []
            st.rerun()
    else:
        st.caption("Click **Save Book** on any recommendation to build your list.")

# --------------------------------------------------
# MAIN TABS
# --------------------------------------------------
tab_similar, tab_personal, tab_about = st.tabs([
    "🔍 Similar Books",
    "👤 Personalized Picks",
    "ℹ️ How It Works"
])

# === TAB 1: SIMILAR BOOKS ===
with tab_similar:
    st.subheader("Find Books Similar to One You Love")

    # Quick-pick sample buttons for instant exploration
    st.caption("Try a quick search:")
    q_cols = st.columns(4)
    samples = ["Harry Potter", "The Hobbit", "1984", "Pride and Prejudice"]
    for i, sample in enumerate(samples):
        if q_cols[i].button(sample, key=f"sample_{i}", use_container_width=True):
            st.session_state.prefill_book = sample

    default_query = st.session_state.pop("prefill_book", "")

    with st.form("similar_books_form"):
        col_input, col_btn = st.columns([4, 1])
        with col_input:
            book_title = st.text_input(
                "Enter a book title",
                value=default_query,
                placeholder="e.g., The Da Vinci Code",
                label_visibility="collapsed"
            )
        with col_btn:
            submitted = st.form_submit_button("Search", type="primary", use_container_width=True)

    if submitted or default_query:
        query = book_title.strip()
        if query:
            if query not in st.session_state.search_history:
                st.session_state.search_history.append(query)

            with st.spinner(f"Finding top {num_recs} books similar to '{query}'..."):
                st.session_state.similar_results = get_similar_books(query, n=num_recs)
        else:
            st.warning("Please enter a book title.")

    # Display persisted results + live filter
    results = st.session_state.similar_results
    if results is not None:
        if not results.empty:
            filter_text = st.text_input("Filter results by keyword or author", key="sim_filter")
            filtered_df = results
            if filter_text:
                mask = results.apply(lambda r: r.astype(str).str.contains(filter_text, case=False).any(), axis=1)
                filtered_df = results[mask]

            st.success(f"Showing {len(filtered_df)} of {len(results)} recommendations")

            if view_mode == "Interactive Cards":
                render_book_cards(filtered_df, key_prefix="sim")
            else:
                st.dataframe(filtered_df, use_container_width=True, hide_index=True)
        else:
            st.warning("No similar books found. Try another title or check spelling.")

# === TAB 2: PERSONALIZED RECOMMENDATIONS ===
with tab_personal:
    st.subheader("Recommendations Tailored to Your Rating History")

    with st.form("personal_form"):
        col_u, col_b = st.columns([3, 1])
        with col_u:
            user_id = st.number_input("Enter User ID", min_value=1, step=1, value=276729)
        with col_b:
            st.write("")
            st.write("")
            rec_submitted = st.form_submit_button("Get Picks", type="primary", use_container_width=True)

    if rec_submitted:
        with st.spinner("Analyzing user rating patterns..."):
            st.session_state.personal_results = recommend(int(user_id), n=num_recs)

    p_results = st.session_state.personal_results
    if p_results is not None:
        if not p_results.empty:
            st.success(f"Found {len(p_results)} personalized recommendations for User #{int(user_id)}")

            if view_mode == "Interactive Cards":
                render_book_cards(p_results, key_prefix="pers")
            else:
                st.dataframe(p_results, use_container_width=True, hide_index=True)
        else:
            st.warning("No recommendations available for this User ID. Try another ID.")

# === TAB 3: ABOUT ===git clone https://github.com/Aditya11495/data-mining-assignment-2-_MSD25020.git
with tab_about:
    st.subheader("How This Recommender Works")
    st.write("""
    This app uses **Item-Based Collaborative Filtering** to suggest books based on shared reader rating patterns rather than genres or keywords alone.
    """)
    c1, c2, c3 = st.columns(3)
    c1.metric("Algorithm", "Cosine Similarity")
    c2.metric("Matrix Format", "SciPy Sparse")
    c3.metric("Frontend", "Streamlit")

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.divider()
st.caption("Book Recommendation System | Built with Python, Collaborative Filtering & Streamlit")
