import streamlit as st
import pickle
import faiss
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from collections import defaultdict

# ── Cấu hình trang ──
st.set_page_config(page_title="Hệ Thống Tìm Kiếm Phim", page_icon="🎬", layout="wide")

# ── 1. Tải hệ thống Index ──
# Dùng st.cache_resource để chỉ load file 1 lần duy nhất khi mở web
@st.cache_resource
def load_system():
    # Load BM25
    with open("index/bm25_index.pkl", "rb") as f:
        bm25_data = pickle.load(f)
        bm25 = bm25_data["bm25"]
    
    # Load FAISS
    cpu_index = faiss.read_index("index/faiss_index.bin")
    
    # Load Metadata
    with open("index/movie_meta.pkl", "rb") as f:
        meta = pickle.load(f)
        
    # Load Embedding Model (Tự động chạy trên CPU ở máy local)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    
    return bm25, cpu_index, meta, model

bm25, cpu_index, meta, model = load_system()

# ── 2. Logic Tokenizer ──
def tokenize_with_prefix(text: str) -> list:
    if not isinstance(text, str): return []
    words = text.lower().split()
    tokens = []
    for word in words:
        tokens.append(word)
        if len(word) >= 6:
            tokens.append(word[:5])
    return tokens

# ── 3. Logic Tìm kiếm & Bộ Lọc ──
def search_movies(query, mode="Hybrid", top_k=10, alpha=0.5, genre="All", year_range=(1970, 2024), min_rating=0.0):
    # BM25 Search
    bm25_query_tokens = tokenize_with_prefix(query)
    query_compact = re.sub(r"[^\w]", "", query.lower())
    if query_compact and query_compact not in bm25_query_tokens:
        bm25_query_tokens.append(query_compact)
        
    bm25_scores = bm25.get_scores(bm25_query_tokens)
    bm25_top_ids = np.argsort(bm25_scores)[::-1][:200]
    
    # Semantic Search
    q_vec = model.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    sem_scores, sem_ids = cpu_index.search(q_vec, 200)
    
    # Fusion (RRF)
    scores = defaultdict(lambda: {"score": 0.0, "bm25_rank": "N/A", "sem_rank": "N/A"})
    k_rrf = 60
    
    if mode in ["BM25", "Hybrid"]:
        for rank, row_id in enumerate(bm25_top_ids):
            scores[row_id]["bm25_rank"] = rank + 1
            if mode == "BM25": scores[row_id]["score"] += 1.0 / (rank + 1)
            else: scores[row_id]["score"] += alpha * (1.0 / (k_rrf + rank + 1))
            
    if mode in ["Semantic", "Hybrid"]:
        for rank, row_id in enumerate(sem_ids[0]):
            if row_id != -1:
                scores[row_id]["sem_rank"] = rank + 1
                if mode == "Semantic": scores[row_id]["score"] += 1.0 / (rank + 1)
                else: scores[row_id]["score"] += (1 - alpha) * (1.0 / (k_rrf + rank + 1))
                
    merged = sorted([{"row_id": int(rid), **info} for rid, info in scores.items()], key=lambda x: x["score"], reverse=True)
    
    # Lọc kết quả (Filter)
    results = []
    for r in merged:
        m = meta[r["row_id"]]
        
        # Lọc Rating
        if float(m.get("vote_average", 0)) < min_rating: continue
        
        # Lọc Năm
        year = str(m.get("release_date", ""))[:4]
        if not year.isdigit() or not (year_range[0] <= int(year) <= year_range[1]): continue
        
        # Lọc Thể loại
        if genre != "All" and genre.lower() not in str(m.get("genres", "")).lower(): continue
        
        results.append(r)
        if len(results) >= top_k: break
        
    return results

# ── 4. Giao diện UI ──
st.title("🎬 Hệ thống Tìm kiếm Phim (Hybrid Search)")
st.markdown("Xây dựng với BM25, Sentence-Transformers và FAISS")

# Khởi tạo danh sách thể loại động từ Metadata
all_genres = set()
for m in meta:
    for g in str(m.get("genres", "")).split():
        if g: all_genres.add(g.replace("_", " ").title())
genre_list = ["All"] + sorted(list(all_genres))

# Chia cột giao diện
col1, col2 = st.columns([1, 3])

with col1:
    st.header("⚙️ Tùy chỉnh")
    mode = st.radio("Chế độ tìm kiếm:", ["Hybrid", "BM25", "Semantic"])
    
    st.markdown("---")
    st.subheader("Bộ lọc (Filters)")
    genre = st.selectbox("Thể loại:", genre_list)
    year_range = st.slider("Năm phát hành:", 1970, 2024, (1990, 2024))
    min_rating = st.slider("Rating tối thiểu:", 0.0, 10.0, 0.0, 0.5)

with col2:
    query = st.text_input("🔍 Nhập mô tả hoặc tên phim...", placeholder="VD: superhero saves the world from alien invasion")
    
    if query:
        with st.spinner("Đang xử lý truy vấn..."):
            results = search_movies(query, mode, top_k=10, genre=genre, year_range=year_range, min_rating=min_rating)
            
            if not results:
                st.warning("Không tìm thấy phim phù hợp với bộ lọc!")
            
            for i, res in enumerate(results):
                m = meta[res["row_id"]]
                year = str(m.get("release_date", ""))[:4]
                rating = m.get("vote_average", 0)
                
                with st.expander(f"[{i+1}] {m['title']} ({year}) - ⭐ {rating:.1f}", expanded=(i==0)):
                    # Hiển thị điểm số chi tiết để demo sự khác biệt giữa các mô hình
                    st.caption(f"**Score:** {res['score']:.4f} | **Rank BM25:** {res['bm25_rank']} | **Rank Semantic:** {res['sem_rank']}")
                    st.write(f"**Thể loại:** {str(m.get('genres', '')).replace('_', ' ').title()}")
                    st.write(f"**Nội dung:** {m.get('overview', '')}")