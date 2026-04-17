import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
import pickle
import os

class MovieSearchEngine:
    def __init__(self, data_path='movies.csv', embedding_path='movie_embeddings.pkl'):
        self.data_path = data_path
        self.embedding_path = embedding_path
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.df = None
        self.embeddings = None
        self._load_data()

    def _load_data(self):
        # Tải dữ liệu text
        self.df = pd.read_csv(self.data_path)
        self.df['overview'] = self.df['overview'].fillna('')

        # Tải hoặc tạo dữ liệu vector
        if os.path.exists(self.embedding_path):
            with open(self.embedding_path, 'rb') as f:
                self.embeddings = pickle.load(f)
        else:
            descriptions = self.df['overview'].tolist()
            self.embeddings = self.model.encode(descriptions, convert_to_tensor=True)
            with open(self.embedding_path, 'wb') as f:
                pickle.dump(self.embeddings, f)

    def search(self, query, top_n=5):
        """Hàm trả về danh sách Dictionary để dễ dàng xử lý trong App"""
        query_vec = self.model.encode(query, convert_to_tensor=True)
        cosine_scores = util.cos_sim(query_vec, self.embeddings)[0]
        
        # Lấy top N
        top_results = torch.topk(cosine_scores, k=top_n)
        
        results = []
        for score, idx in zip(top_results[0], top_results[1]):
            item = self.df.iloc[idx.item()]
            results.append({
                "name": item['name'],
                "overview": item['overview'],
                "score": round(score.item(), 4)
            })
        return results