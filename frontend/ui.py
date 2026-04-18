import streamlit as st

def load_css():
    with open('frontend/style.css', 'r', encoding='utf-8') as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

def render_header():
    st.markdown("""
        <div style="text-align: center; padding: 40px 0 50px 0;">
            <h1 class="main-title" style="font-size: 4rem; margin-bottom: 10px;">
                MOVIE RETRIEVAL
            </h1>
        </div>
    """, unsafe_allow_html=True)

def render_movie_card(movie_row, rank=None, score=None, algo=None, is_random=False):
    title = movie_row.get('name', 'Unknown Title')
    overview = movie_row.get('overview', 'No description available.')
    rating = movie_row.get('vote_average', 0.0)
    year = movie_row.get('year', 'N/A')
    genres = movie_row.get('genre_list', [])
    movie_id = str(movie_row.get('id', ''))
    
    # 2. Xây dựng các Badge hiển thị (Loại bỏ emoji)
    badges_html = ""
    if rank is not None:
        badges_html += f'<span class="badge badge-rank">Order {rank}</span>'
    
    badges_html += f'<span class="badge badge-star">Rating {rating}</span>'
    
    for g in genres[:2]:
        badges_html += f'<span class="badge badge-genre">{g}</span>'

    # Khối thẻ HTML Card Vertical
    html = f"""
    <div class="movie-card">
        <div class="movie-poster">
            <div class="poster-letter">{title[0] if title else '?'}</div>
        </div>
        <div class="movie-info">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <span style="font-size: 0.85rem; font-weight: 600; color: #94a3b8; background: #f1f5f9; padding: 2px 8px; border-radius: 6px;">{year}</span>
            </div>
            <h3 class="movie-title">{title}</h3>
            <div class="movie-badges">{badges_html}</div>
            <p class="movie-overview">{overview}</p>
        </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)
    
    # Nút bấm Xem Chi tiết (Loại bỏ emoji)
    if st.button(f"Details", key=f"btn_detail_{movie_id}_{is_random}", use_container_width=True):
        st.session_state.selected_movie_id = movie_id
        st.rerun()

def render_movie_detail_page(movie_row):
    title = movie_row.get('name', 'Unknown')
    overview = movie_row.get('overview', '')
    genres = ", ".join(movie_row.get('genre_list', []))
    year = movie_row.get('year', 'N/A')
    rating = movie_row.get('vote_average', 0.0)
    director = movie_row.get('director', 'Unknown')
    cast = ", ".join(movie_row.get('cast_names', []))
    tagline = movie_row.get('tagline', '')
    budget = movie_row.get('budget', 0)
    revenue = movie_row.get('revenue', 0)

    # UI Chi tiết (Loại bỏ emoji)
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown(f"""
            <div class="detail-poster" style="height: 500px; border-radius: 32px; display: flex; 
                 align-items: center; justify-content: center; font-size: 180px; font-weight: 800; color: #64748b;
                 border: 1px solid rgba(226, 232, 240, 0.8); position: sticky; top: 2rem;">
                <span class="main-title">{title[0]}</span>
            </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.title(title)
        if tagline:
            st.markdown(f"*'{tagline}'*")
        
        st.markdown(f"**Year:** {year} | **Rating:** {rating}/10")
        st.markdown(f"**Genres:** {genres}")
        st.markdown("---")
        st.write(overview)
        st.markdown("---")
        
        d_col1, d_col2 = st.columns(2)
        d_col1.markdown(f"**Director:** {director}")
        d_col2.markdown(f"**Cast:** {cast}")
        
        st.markdown(f"**Budget:** \\${budget:,} | **Revenue:** \\${revenue:,}")
