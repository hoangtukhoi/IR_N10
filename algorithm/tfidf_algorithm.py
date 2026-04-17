import math
import re
from collections import Counter, defaultdict
import nltk
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)

class TFIDF:
    def __init__(self, corpus_raw, language='english'):
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words(language))
        
        # Preprocess corpus
        self.corpus = [self._preprocess(doc) for doc in corpus_raw]
        self.corpus_size = len(self.corpus)
        
        if self.corpus_size == 0:
            raise ValueError("Corpus rỗng")
        
        # Build document frequencies and inverted index
        self.df = defaultdict(int)
        self.inverted_index = defaultdict(list)
        
        for i, doc in enumerate(self.corpus):
            freq = Counter(doc)
            # tf: term count in the document
            for word, count in freq.items():
                self.df[word] += 1
                self.inverted_index[word].append((i, count))
                
        # Calculate IDF
        self.idf = {}
        for word, df in self.df.items():
            self.idf[word] = math.log((1 + self.corpus_size) / (1 + df)) + 1

    def _preprocess(self, text):
        text = str(text).lower()
        text = re.sub(r'[^\w\s]', '', text)
        tokens = text.split()
        tokens = [w for w in tokens if w not in self.stop_words and len(w) > 1]
        return [self.stemmer.stem(w) for w in tokens]
    
    def get_scores(self, query_raw):
        query = self._preprocess(query_raw)
        scores = defaultdict(float)
        
        for q in query:
            if q not in self.inverted_index:
                continue
            idf = self.idf[q]
            # TF-IDF of the query term: let's assume query term format count is 1 for simplicity
            for doc_id, f in self.inverted_index[q]:
                # tf is just the count of term in doc. Can also normalize by doc length.
                # let's use standard term frequency
                tf = f 
                scores[doc_id] += tf * idf
                
        # Normalize the scores or keep them as raw dot products. Raw dot products are fine for ranking.
        return [scores.get(i, 0.0) for i in range(self.corpus_size)]
    
    def get_top_n(self, query_raw, n=10):
        scores = self.get_scores(query_raw)
        return sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[:n]
