import math
import re
from collections import Counter
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)
    tokens = text.split()
    return [word for word in tokens if word not in ENGLISH_STOP_WORDS]

class SimpleBM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        self.corpus_size = len(corpus)
        self.avgdl = sum(len(doc) for doc in corpus) / self.corpus_size if self.corpus_size > 0 else 0
        self.corpus = corpus
        self.k1 = k1
        self.b = b
        
        self.df = {}
        self.idf = {}
        self.doc_freqs = []
        
        for doc in corpus:
            freq = Counter(doc)
            self.doc_freqs.append(freq)
            for word in freq.keys():
                self.df[word] = self.df.get(word, 0) + 1
                
        for word, freq in self.df.items():
            self.idf[word] = math.log(((self.corpus_size - freq + 0.5) / (freq + 0.5)) + 1)
            
    def get_scores(self, query):
        scores = [0.0] * self.corpus_size
        for q in query:
            if q not in self.df:
                continue
            idf = self.idf[q]
            for i, doc_freq in enumerate(self.doc_freqs):
                if q in doc_freq:
                    f = doc_freq[q]
                    doc_len = len(self.corpus[i])
                    score = idf * (f * (self.k1 + 1)) / (f + self.k1 * (1 - self.b + self.b * (doc_len / self.avgdl)))
                    scores[i] += score
        return scores