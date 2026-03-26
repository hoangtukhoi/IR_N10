# TMDB Movie Search Engine — Hybrid Information Retrieval

Hệ thống tìm kiếm phim theo mô tả văn bản, kết hợp giữa lexical search (BM25) và semantic search (vector embedding). Mục tiêu là cải thiện độ chính xác trong bài toán Information Retrieval.

## Features

* Hybrid search (RRF): kết hợp BM25 và FAISS với trọng số tùy chỉnh
* Typo tolerance: custom tokenizer (prefix 5 ký tự) giúp xử lý sai chính tả và từ dính
* Indexing: tăng trọng số title x4 để ưu tiên tên phim
* Filtering: lọc theo genre, năm phát hành, IMDb
* UI: Streamlit, hỗ trợ so sánh BM25 / Semantic / Hybrid

## Project Structure

```
movie-hybrid-search/
├── tmdb_hybrid_search_training.ipynb
├── app.py
├── requirements.txt
├── README.md
└── index/
    ├── bm25_index.pkl
    ├── faiss_index.bin
    └── movie_meta.pkl
```

## Reproduction

### Step 1: Training (Kaggle)

* Dataset: asaniczka/tmdb-movies-dataset-2023-930k-movies
* Bật GPU (T4/P100)
* Chạy `tmdb_hybrid_search_training.ipynb`
* Tải về:

```
bm25_index.pkl
faiss_index.bin
movie_meta.pkl
```

### Step 2: Run local

* Đặt file vào thư mục `index/`
* Cài đặt:

```
pip install -r requirements.txt
```

* Chạy:

```
streamlit run app.py
```

* Truy cập: [http://localhost:8501](http://localhost:8501)

## Evaluation

* Precision@5, Precision@10
* Mean Reciprocal Rank (MRR)
* Hybrid cho kết quả tốt hơn BM25 và Semantic riêng lẻ

## Tech Stack

* pandas, regex
* rank-bm25
* sentence-transformers (all-MiniLM-L6-v2)
* faiss
* streamlit
