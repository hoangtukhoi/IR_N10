import numpy as np
import faiss
from collections import defaultdict
from sentence_transformers import SentenceTransformer

class HybridAlgorithm:
    def __init__(self, corpus_raw, bm25_algorithm, alpha=0.5):
        """
        Khởi tạo Hybrid Algorithm kết hợp BM25 và Semantic Search (FAISS).
        - corpus_raw: Danh sách các văn bản (ví dụ: overview của các bộ phim).
        - bm25_algorithm: Đối tượng BM25 đã được khởi tạo để dùng lại chỉ số của nó.
        - alpha: Hệ số kết hợp RRF (Reciprocal Rank Fusion). alpha cho BM25 và (1-alpha) cho Semantic.
        """
        self.bm25 = bm25_algorithm
        self.corpus_size = len(corpus_raw)
        self.alpha = alpha
        
        if self.corpus_size == 0:
            raise ValueError("Corpus rỗng")
        
        # Load mô hình nhúng (Embedding Model) phù hợp chạy nhanh trên CPU
        self.model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        
        # Encode corpus
        embeddings = self.model.encode(
            corpus_raw, 
            batch_size=256, 
            show_progress_bar=False, 
            normalize_embeddings=True, 
            convert_to_numpy=True
        ).astype("float32")
        
        # Build FAISS index for Maximum Inner Product Search (Cosine Similarity vì đã normalize)
        dim = embeddings.shape[1]
        self.cpu_index = faiss.IndexFlatIP(dim)
        self.cpu_index.add(embeddings)
        
    def get_scores(self, query):
        """
        Tính điểm kết hợp (Hybrid Score) cho query.
        """
        scores = defaultdict(float)
        
        # 1. Điểm BM25
        # Sử dụng lại thuật toán BM25 gốc có trong hệ thống
        bm25_scores = self.bm25.get_scores(query)
        # Giới hạn top 100 của BM25 để tối ưu RRF
        bm25_top_ids = np.argsort(bm25_scores)[::-1][:100]
        
        # 2. Điểm Semantic (Sentence Transformers)
        q_vec = self.model.encode([query], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
        # Tìm top 100 từ faiss index
        sem_scores, sem_ids = self.cpu_index.search(q_vec, 100)
        
        # 3. RRF Fusion (Reciprocal Rank Fusion)
        k_rrf = 60
        
        # Cộng điểm RRF từ BM25
        for rank, row_id in enumerate(bm25_top_ids):
            scores[row_id] += self.alpha * (1.0 / (k_rrf + rank + 1))
            
        # Cộng điểm RRF từ Semantic Search
        for rank, row_id in enumerate(sem_ids[0]):
            if row_id != -1:
                scores[row_id] += (1 - self.alpha) * (1.0 / (k_rrf + rank + 1))
                
        # Trả về kết quả song song dưới dạng list như BM25 và TFIDF
        return [scores.get(i, 0.0) for i in range(self.corpus_size)]
