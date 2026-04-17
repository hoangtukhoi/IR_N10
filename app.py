import streamlit as st
import pandas as pd
from algorithm.bm25_algorithm import BM25
from algorithm.tfidf_algorithm import TFIDF
from algorithm.hybrid_algorithm import HybridAlgorithm

# Cấu hình trang cơ bản
st.set_page_config(page_title="Movie Retrieval", layout="centered")

# Hàm load dữ liệu và khởi tạo mô hình (được cache để chỉ chạy 1 lần duy nhất)
@st.cache_resource
def load_system():
    # Load dữ liệu từ thư mục data
    df = pd.read_csv('data/movies.csv')
    df['overview'] = df['overview'].fillna('')
    
    # Tiền xử lý tập dữ liệu
    tokenized_corpus = df['overview'].tolist()
    
    # Khởi tạo mô hình
    bm25 = BM25(tokenized_corpus)
    tfidf = TFIDF(tokenized_corpus)
    hybrid = HybridAlgorithm(tokenized_corpus, bm25)
    
    return df, bm25, tfidf, hybrid

# Chạy hàm load
df, bm25, tfidf, hybrid = load_system()

# --- XÂY DỰNG GIAO DIỆN STREAMLIT ---


# Ô nhập liệu
query = st.text_input("Nhập từ khóa tiếng Anh (VD: alien planet space):")
top_n = st.slider("Số lượng kết quả hiển thị:", min_value=1, max_value=10, value=3)
algo_choice = st.selectbox("Chọn thuật toán:", ("Hybrid", "BM25", "TF-IDF"))

# Nút bấm tìm kiếm
if st.button("Tìm kiếm"):
    if query.strip() == "":
        st.warning("Vui lòng nhập từ khóa tìm kiếm!")
    else:
        # Xử lý truy vấn
        if algo_choice == "BM25":
            scores = bm25.get_scores(query)
        elif algo_choice == "TF-IDF":
            scores = tfidf.get_scores(query)
        else:
            # Hybrid
            scores = hybrid.get_scores(query)
        
        # Sắp xếp và lấy top kết quả
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_n]
        
        # Hiển thị kết quả
        st.subheader("Kết quả tìm kiếm:")
        for rank, idx in enumerate(top_indices, 1):
            score = scores[idx]
            movie_name = df.iloc[idx]['name']
            overview = df.iloc[idx]['overview']
            
            # Khung hiển thị đơn giản
            with st.container():
                st.markdown(f"**Top {rank}: {movie_name}** *(Điểm {algo_choice}: {score:.2f})*")
                st.write(overview)
                st.divider() # Đường kẻ ngang phân cách