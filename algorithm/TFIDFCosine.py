import math
import re
from collections import defaultdict


STOPWORDS = {
    "a", "an", "the", "is", "it", "in", "on", "at", "to", "for", "of",
    "and", "or", "but", "with", "by", "from", "as", "be", "was", "were",
    "are", "has", "have", "had", "been", "his", "her", "their", "its",
    "he", "she", "they", "we", "i", "my", "our", "who", "what", "when",
    "where", "which", "that", "this", "these", "those", "not", "no",
    "can", "will", "would", "could", "should", "may", "might", "must",
    "do", "did", "does", "into", "up", "out", "about", "after", "before",
    "over", "under", "between", "through", "during", "without", "within",
    "against", "around", "among", "each", "every", "all", "both", "few",
    "more", "most", "some", "any", "other", "another", "new", "just",
    "also", "only", "so", "then", "than", "too", "very", "even", "still",
    "back", "down", "off", "again", "while", "however", "although",
    "because", "since", "if", "though", "seem", "seems", "become",
    "make", "take", "give", "find", "come", "go", "see", "know", "get",
    "use", "try", "one", "two", "three", "four", "five",
}


class TFIDFCosine:
    def __init__(self, corpus_raw, min_token_len=2):
        self._stopwords = STOPWORDS
        self._min_len = min_token_len

        # Preprocess corpus
        self.corpus_size = len(corpus_raw)

        if self.corpus_size == 0:
            raise ValueError("Corpus rỗng")

        # Tokenize toàn bộ corpus
        all_tokens = [self._tokenize(doc) for doc in corpus_raw]

        # Tính IDF
        self._idf = self._compute_idf(all_tokens)

        # Tính sẵn TF-IDF vector cho mỗi document (chỉ tính 1 lần)
        self._doc_vectors = [
            self._tfidf_vector(self._compute_tf(tokens))
            for tokens in all_tokens
        ]

        # Tính sẵn norm cho mỗi document vector (tối ưu tốc độ search)
        self._doc_norms = [
            math.sqrt(sum(v ** 2 for v in vec.values())) if vec else 0.0
            for vec in self._doc_vectors
        ]

    # ── Tiền xử lý

    def _tokenize(self, text):

        text = str(text).lower()
        text = re.sub(r"[^a-z\s]", " ", text)
        return [
            t for t in text.split()
            if t not in self._stopwords and len(t) > self._min_len
        ]

    # ── TF

    def _compute_tf(self, tokens):

        tf = defaultdict(int)
        for t in tokens:
            tf[t] += 1
        total = len(tokens) or 1
        return {word: count / total for word, count in tf.items()}

    # ── IDF 

    def _compute_idf(self, doc_tokens):

        N = len(doc_tokens)
        df = defaultdict(int)
        for tokens in doc_tokens:
            for word in set(tokens):
                df[word] += 1
        return {
            word: math.log((N + 1) / (freq + 1)) + 1
            for word, freq in df.items()
        }

    # ── TF-IDF vector 

    def _tfidf_vector(self, tf):

        return {
            word: tf_val * self._idf.get(word, 0)
            for word, tf_val in tf.items()
        }

    # ── Cosine Similarity 

    def _cosine_similarity(self, vec_a, norm_a, vec_b, norm_b):

        if norm_a == 0 or norm_b == 0:
            return 0.0

        # Chỉ tính dot product trên các từ chung
        # Duyệt vector nhỏ hơn để tối ưu
        if len(vec_a) > len(vec_b):
            vec_a, vec_b = vec_b, vec_a

        dot = 0.0
        for word, val in vec_a.items():
            if word in vec_b:
                dot += val * vec_b[word]

        if dot == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    # ── Public API 

    def get_scores(self, query_raw):
        q_tokens = self._tokenize(query_raw)
        if not q_tokens:
            return [0.0] * self.corpus_size

        q_tf = self._compute_tf(q_tokens)
        q_vec = self._tfidf_vector(q_tf)
        q_norm = math.sqrt(sum(v ** 2 for v in q_vec.values()))

        if q_norm == 0:
            return [0.0] * self.corpus_size

        scores = []
        for i in range(self.corpus_size):
            score = self._cosine_similarity(
                q_vec, q_norm,
                self._doc_vectors[i], self._doc_norms[i]
            )
            scores.append(score)

        return scores

    def get_top_n(self, query_raw, n=10):
        """Trả về top n kết quả tốt nhất."""
        scores = self.get_scores(query_raw)
        return sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:n]