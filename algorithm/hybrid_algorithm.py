from collections import Counter

class HybridSearch:
    def __init__(self, bm25_model, tfidf_model, k=60):
        """
        Khởi tạo hệ thống Hybrid Search bằng mô hình BM25 và TF-IDF đã huấn luyện.
        Sử dụng thuật toán Reciprocal Rank Fusion (RRF).
        Tham số k=60 là thông số chuẩn theo bài báo gốc của RRF.
        """
        self.bm25 = bm25_model
        self.tfidf = tfidf_model
        # Lấy kích thước tập dữ liệu
        self.corpus_size = self.bm25.corpus_size
        self.k = k

    def get_scores(self, query):
        """
        Lấy điểm sau khi kết hợp kết quả.
        """
        bm25_scores = self.bm25.get_scores(query)
        tfidf_scores = self.tfidf.get_scores(query)

        # Mảng lưu điểm tổng (đã scale theo RRF)
        hybrid_scores = [0.0] * self.corpus_size

        # 1. Tính toán thứ hạng và điểm RRF cho BM25
        # Lọc ra các document có điểm > 0 và sắp xếp
        bm25_sorted_indices = [
            i for i, v in sorted(enumerate(bm25_scores), key=lambda x: x[1], reverse=True) 
            if v > 0
        ]
        for rank, doc_id in enumerate(bm25_sorted_indices):
            # Điểm = 1 / (k + hạng của document)
            hybrid_scores[doc_id] += 1.0 / (self.k + rank + 1)

        # 2. Tính toán thứ hạng và điểm RRF cho TF-IDF
        # Lọc ra các document có điểm > 0 và sắp xếp
        tfidf_sorted_indices = [
            i for i, v in sorted(enumerate(tfidf_scores), key=lambda x: x[1], reverse=True) 
            if v > 0
        ]
        for rank, doc_id in enumerate(tfidf_sorted_indices):
            hybrid_scores[doc_id] += 1.0 / (self.k + rank + 1)

        return hybrid_scores

    def get_top_n(self, query, n=10):
        """
        Trả về top n kết quả tốt nhất.
        """
        scores = self.get_scores(query)
        # Sắp xếp để lấy rank
        return sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:n]

    def get_scores_with_prf(self, query, top_n_docs=3, top_k_terms=3):
        """
        Tìm kiếm với tính năng Pseudo-Relevance Feedback (PRF - Mở rộng truy vấn).
        - query: câu hỏi gốc
        - top_n_docs: lấy bao nhiêu tài liệu top đầu để phân tích
        - top_k_terms: lấy bao nhiêu từ khóa phổ biến nhất để mở rộng
        """
        # 1. Chạy tìm kiếm lần 1 để lấy thứ hạng ban đầu
        initial_scores = self.get_scores(query)
        
        # 2. Lấy danh sách ID của Top N tài liệu tốt nhất
        valid_indices = [i for i, v in enumerate(initial_scores) if v > 0]
        top_docs_indices = sorted(valid_indices, key=lambda i: initial_scores[i], reverse=True)[:top_n_docs]
        
        if not top_docs_indices:
            self.last_expanded_query = query
            return initial_scores

        # 3. Trích xuất tất cả từ vựng từ top N tài liệu này
        query_tokens = set(self.bm25._preprocess(query))
        expanded_terms = []
        
        for doc_id in top_docs_indices:
            doc_tokens = self.bm25.corpus[doc_id] # Lấy danh sách từ (đã preprocess) của bộ phim
            for token in doc_tokens:
                # Bỏ qua từ nếu nó đã nằm trong câu truy vấn gốc
                if token not in query_tokens:
                    expanded_terms.append(token)
                    
        # 4. Tìm Top K từ khóa thông dụng nhất
        most_common = [term for term, count in Counter(expanded_terms).most_common(top_k_terms)]
        
        # 5. Gộp câu query cũ và các từ mới được kết xuất
        new_query = query + " " + " ".join(most_common)
        
        # Lưu lại để hiển thị cho hệ thống thay vì in ngầm
        self.last_expanded_query = new_query
        
        # 6. Chạy tìm kiếm lần 2 với câu query đã được nâng cấp (Bản chất PRF)
        return self.get_scores(new_query)
