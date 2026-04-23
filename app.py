import streamlit as st
import pandas as pd
from algorithm.bm25_algorithm import BM25
from algorithm.tfidf_algorithm import TFIDF
from algorithm.TFIDFCosine import TFIDFCosine
from algorithm.hybrid_algorithm import HybridSearch
from algorithm.spell_checker import SpellChecker
from eval.evaluation import Evaluation
import time

# Cấu hình trang cơ bản
st.set_page_config(page_title="Movie Retrieval Engine", page_icon="M", layout="wide", initial_sidebar_state="expanded")

# Tải giao diện Frontend
import frontend.ui as ui
ui.load_css()
ui.render_header()

# Hàm load dữ liệu và khởi tạo mô hình
@st.cache_resource
def load_system():
    df_movies = pd.read_csv('data/tmdb_5000_movies.csv')
    df_credits = pd.read_csv('data/tmdb_5000_credits.csv')
    df = pd.merge(df_movies, df_credits, left_on='id', right_on='movie_id')
    df = df.rename(columns={'title_x': 'name'}) 
    df['overview'] = df['overview'].fillna('')
    
    import json
    def parse_json_col(x, key_name=None):
        try:
            data = json.loads(x)
            if key_name: return [i[key_name] for i in data]
            return data
        except: return []

    df['genre_list'] = df['genres'].apply(lambda x: parse_json_col(x, 'name'))
    df['cast_names'] = df['cast'].apply(lambda x: parse_json_col(x, 'name')[:5])
    
    def get_director(x):
        crew = parse_json_col(x)
        for c in crew:
            if c.get('job') == 'Director': return c.get('name')
        return 'Unknown'
        
    df['director'] = df['crew'].apply(get_director)
    df['year'] = pd.to_datetime(df['release_date'], errors='coerce').dt.year.fillna(0).astype(int)
    
    tokenized_corpus = df['overview'].tolist()
    bm25 = BM25(tokenized_corpus)
    tfidf = TFIDF(tokenized_corpus)
    tfidf_cosine = TFIDFCosine(tokenized_corpus)
    hybrid = HybridSearch(bm25, tfidf)
    
    corpus_texts = df['overview'].tolist() + df['name'].tolist()
    spell_checker = SpellChecker(corpus_texts)
    
    evaluator = Evaluation(df=df)
    
    return df, bm25, tfidf, tfidf_cosine, hybrid, spell_checker, evaluator

df, bm25, tfidf, tfidf_cosine, hybrid, spell_checker, evaluator = load_system()

# --- QUẢN LÝ SESSION STATE ---
if 'selected_movie_id' not in st.session_state: st.session_state.selected_movie_id = None
if 'current_page' not in st.session_state: st.session_state.current_page = 1
if 'search_results' not in st.session_state: st.session_state.search_results = None
if 'search_scores' not in st.session_state: st.session_state.search_scores = None
if 'applied_algo' not in st.session_state: st.session_state.applied_algo = ""
if 'search_query_state' not in st.session_state: st.session_state.search_query_state = ""

# --- ROUTING LOGIC ---

if st.session_state.selected_movie_id is not None:
    if st.button("Back to Home"):
        st.session_state.selected_movie_id = None
        st.rerun()
        
    target_id = st.session_state.selected_movie_id
    movie_row = df[df['id'].astype(str) == str(target_id)]
    
    if not movie_row.empty:
        ui.render_movie_detail_page(movie_row.iloc[0])
    else:
        st.error(f"Error: Movie ID {target_id} not found.")
        if st.button("Return"):
            st.session_state.selected_movie_id = None
            st.rerun()

else:
    def trigger_search_callback():
        st.session_state.current_page = 1

    search_query = st.text_input(
        "Search movies by title or description...", 
        value=st.session_state.search_query_state,
        key="main_search_input"
    )
    st.session_state.search_query_state = search_query

    st.markdown('<div class="filter-container">', unsafe_allow_html=True)
    fcol1, fcol2, fcol3, fcol4, fcol5 = st.columns([1.5, 2, 1.5, 1, 1])
    
    algo_choice = fcol1.selectbox("Algorithm:", ("BM25", "TF-IDF", "TF-IDF Cosine", "Hybrid (BM25 + TF-IDF)", "Hybrid + PRF"))
    all_genres = set(g for sublist in df['genre_list'] for g in sublist)
    selected_genres = fcol2.multiselect("Genres:", sorted(list(all_genres)))
    min_year, max_year = fcol3.slider("Year:", 1920, 2024, (1990, 2024))
    min_rating = fcol4.number_input("Min Rating:", 0.0, 10.0, 0.0, step=0.5)
    top_n_limit = fcol5.number_input("Top Results:", 1, 100, 18)
    st.markdown('</div>', unsafe_allow_html=True)

    # --- BATCH EVALUATION BUTTON ---
    with st.sidebar:
        st.subheader("System Evaluation")
        if st.button("Run Batch Evaluation", use_container_width=True):
            with st.spinner(f"Evaluating {algo_choice}..."):
                # Determine model
                if algo_choice == "BM25": model = bm25
                elif algo_choice == "TF-IDF": model = tfidf
                elif algo_choice == "TF-IDF Cosine": model = tfidf_cosine
                elif algo_choice == "Hybrid (BM25 + TF-IDF)": model = hybrid
                else: model = hybrid # Hybrid + PRF
                
                use_prf = (algo_choice == "Hybrid + PRF")
                
                avg_metrics = evaluator.calculate_batch_metrics(
                    model, df, k=top_n_limit, use_prf=use_prf
                )
                
                if avg_metrics:
                    st.success(f"Batch Evaluation for {algo_choice} completed!")
                    evaluator.print_batch_report(algo_choice, avg_metrics)
                    st.write(avg_metrics)
                    st.info("Check terminal for full report.")
                else:
                    st.error("No ground truth queries found or error during evaluation.")

    btn_col1, btn_col2, _ = st.columns([1, 1, 4])
    perform_search = btn_col1.button("Search", use_container_width=True, on_click=trigger_search_callback)
    
    if algo_choice == "Hybrid + PRF":
        with st.expander("Advanced PRF Settings", expanded=False):
            pcol1, pcol2 = st.columns(2)
            prf_n = pcol1.number_input("Top relevant docs:", 1, 10, 3)
            prf_k = pcol2.number_input("Number of expansion terms:", 1, 10, 3)

    random_pressed = btn_col2.button("Random Shuffle", use_container_width=True)
    if random_pressed:
        st.session_state.search_results = None
        st.session_state.search_query_state = ""
        st.session_state.current_page = 1

    def filter_idx(df_source):
        mask = (df_source['year'] >= min_year) & (df_source['year'] <= max_year) & (df_source['vote_average'] >= min_rating)
        if selected_genres:
            mask = mask & df_source['genre_list'].apply(lambda x: any(g in selected_genres for g in x))
        return set(df_source[mask].index)

    is_trigger = st.session_state.get('trigger_search', False)
    if (perform_search or is_trigger) and search_query.strip() != "":
        st.session_state.trigger_search = False
        corrected = spell_checker.correct_query(search_query)
        if corrected != search_query.lower().strip():
            st.warning(f"Did you mean: {corrected}?")
            def fix_and_rerun():
                st.session_state.main_search_input = corrected
                st.session_state.search_query_state = corrected
                st.session_state.trigger_search = True
                st.session_state.current_page = 1
            st.button(f"Apply correction: {corrected}", on_click=fix_and_rerun)

        start_time = time.time()
        if algo_choice == "BM25": scores = bm25.get_scores(search_query)
        elif algo_choice == "TF-IDF": scores = tfidf.get_scores(search_query)
        elif algo_choice == "TF-IDF Cosine": scores = tfidf_cosine.get_scores(search_query)
        elif algo_choice == "Hybrid (BM25 + TF-IDF)": scores = hybrid.get_scores(search_query)
        else: 
            scores = hybrid.get_scores_with_prf(search_query, prf_n, prf_k)
            st.info(f"Expanded Search Query: {hybrid.last_expanded_query}")
        end_time = time.time()
        exec_time = end_time - start_time

        valid = filter_idx(df)
        indices = [i for i, s in sorted(enumerate(scores), key=lambda x: x[1], reverse=True) if i in valid and s > 0][:top_n_limit]
        
        st.session_state.search_results = indices
        st.session_state.search_scores = scores
        st.session_state.applied_algo = algo_choice
        st.session_state.current_page = 1

        # In báo cáo ra terminal
        retrieved_movie_ids = df.iloc[indices]['id'].tolist()
        evaluator.print_report(algo_choice, search_query, retrieved_movie_ids, exec_time, k=top_n_limit)

    if st.session_state.search_results is not None:
        indices = st.session_state.search_results
        scores = st.session_state.search_scores
        algo = st.session_state.applied_algo
        
        st.subheader(f"Search Results ({len(indices)} movies):")
        
        items_per_page = 6
        total_pages = (len(indices) - 1) // items_per_page + 1
        
        start_idx = (st.session_state.current_page - 1) * items_per_page
        current_batch = indices[start_idx : start_idx + items_per_page]
        
        grid = st.columns(3)
        for i, idx in enumerate(current_batch):
            with grid[i % 3]:
                ui.render_movie_card(df.iloc[idx], rank=start_idx+i+1, score=scores[idx], algo=algo)
        
        st.markdown("<br>", unsafe_allow_html=True)
        p1, p2, p3, p4, p5 = st.columns([2, 1, 2, 1, 2])
        if st.session_state.current_page > 1:
            if p2.button("Previous", use_container_width=True):
                st.session_state.current_page -= 1
                st.rerun()
        
        p3.markdown(f"<div style='text-align:center; font-weight:bold; font-size:1.1rem;'>Page {st.session_state.current_page} / {total_pages}</div>", unsafe_allow_html=True)
        
        if st.session_state.current_page < total_pages:
            if p4.button("Next", use_container_width=True):
                st.session_state.current_page += 1
                st.rerun()

    else:
        st.markdown("---")
        st.subheader("Recommended for you")
        valid = filter_idx(df)
        valid_df = df.iloc[list(valid)]
        
        if not valid_df.empty:
            if 'random_seed_movies' not in st.session_state:
                st.session_state.random_seed_movies = valid_df.sample(n=min(6, len(valid_df)))
            
            random_movies = st.session_state.random_seed_movies
            grid = st.columns(3)
            for i, (_, row) in enumerate(random_movies.iterrows()):
                with grid[i % 3]:
                    ui.render_movie_card(row, rank=i+1, is_random=True)
        else:
            st.info("No movies match the current filters.")

    if 'random_seed_movies' in st.session_state and random_pressed:
         del st.session_state['random_seed_movies']
         st.rerun()