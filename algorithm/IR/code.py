import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import pickle
import os

# --- CẤU HÌNH HỆ THỐNG ---
FILE_DATA = 'movies.csv'
FILE_EMBEDDING = 'movie_embeddings.pkl'
MODEL_NAME = 'all-MiniLM-L6-v2' # Mô hình AI thông minh và nhẹ

def build_search_engine():
    # 1. Tải dữ liệu phim
    print("--- Đang khởi động hệ thống ---")
    df = pd.read_csv(FILE_DATA)
    df['overview'] = df['overview'].fillna('')

    # 2. Tải mô hình AI
    model = SentenceTransformer(MODEL_NAME)

    # 3. Kiểm tra file vector đã lưu chưa
    if os.path.exists(FILE_EMBEDDING):
        print(f"Sẵn sàng! Đang tải dữ liệu vector từ: {FILE_EMBEDDING}")
        with open(FILE_EMBEDDING, 'rb') as f:
            movie_embeddings = pickle.load(f)
    else:
        print("Đang tạo dữ liệu vector lần đầu (quá trình này mất khoảng 30s-1p)...")
        movie_descriptions = df['overview'].tolist()
        # Biến mô tả thành vector số
        movie_embeddings = model.encode(movie_descriptions, convert_to_tensor=True)
        
        # Lưu lại để lần sau dùng luôn
        with open(FILE_EMBEDDING, 'wb') as f:
            pickle.dump(movie_embeddings, f)
        print(f"Đã lưu xong file vector: {FILE_EMBEDDING}")

    return df, model, movie_embeddings

def semantic_search(query, df, model, movie_embeddings, top_n=5):
    # Chuyển câu hỏi của người dùng thành vector
    query_embedding = model.encode(query, convert_to_tensor=True)
    
    # Tính toán độ tương đồng Cosine
    cosine_scores = util.cos_sim(query_embedding, movie_embeddings)[0]
    
    # Lấy top N kết quả cao nhất
    top_results = torch.topk(cosine_scores, k=top_n)
    
    print(f"\n>>> Kết quả cho mô tả: '{query}'")
    print("-" * 50)
    
    for score, idx in zip(top_results[0], top_results[1]):
        movie_name = df.iloc[idx.item()]['name']
        overview = df.iloc[idx.item()]['overview']
        print(f"Phim: {movie_name}")
        print(f"Độ chính xác: {score:.4f}")
        print(f"Mô tả ngắn: {overview[:150]}...")
        print("-" * 50)

# --- CHẠY CHƯƠNG TRÌNH ---
if __name__ == "__main__":
    # Khởi tạo hệ thống
    df, model, movie_embeddings = build_search_engine()

    # Thử nghiệm tìm kiếm
    while True:
        user_query = input("\nNhập mô tả phim bạn muốn tìm (hoặc 'exit' để thoát): ")
        if user_query.lower() == 'exit':
            break
        semantic_search(user_query, df, model, movie_embeddings)