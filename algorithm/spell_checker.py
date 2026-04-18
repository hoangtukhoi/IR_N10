import re
from collections import Counter

class SpellChecker:
    def __init__(self, corpus_texts):
        """
        Khởi tạo từ điển vựng (Vocabulary) từ bộ dữ liệu văn bản thuần.
        """
        self.WORDS = Counter()
        for text in corpus_texts:
            # Chỉ lấy các từ là chữ cái/số, chuyển thành in thường nguyên bản (không qua stemming)
            words = re.findall(r'\w+', str(text).lower())
            self.WORDS.update(words)
        self.N = sum(self.WORDS.values())
        
        # Nếu tổng số từ = 0 (trường hợp lỗi mạng/dữ liệu), phòng hờ chia cho 0
        if self.N == 0:
            self.N = 1

    def P(self, word): 
        """Xác suất xuất hiện của từ (Probability of word)."""
        return self.WORDS[word] / self.N

    def correction(self, word): 
        """Gợi ý từ đúng có xác suất cao nhất."""
        return max(self.candidates(word), key=self.P)

    def candidates(self, word): 
        """Sinh ra các từ có khả năng là bản sửa lỗi đúng."""
        return (self.known([word]) or self.known(self.edits1(word)) or self.known(self.edits2(word)) or [word])

    def known(self, words): 
        """Tập hợp con các từ thực sự xuất hiện trong Từ điển."""
        return set(w for w in words if w in self.WORDS)

    def edits1(self, word):
        """Tất cả các định dạng sai 1 khoảng cách (chữ cái) so với từ gốc."""
        letters    = 'abcdefghijklmnopqrstuvwxyz0123456789'
        splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
        deletes    = [L + R[1:]               for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
        replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
        inserts    = [L + c + R               for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def edits2(self, word): 
        """Tất cả các định dạng sai 2 khoảng cách so với từ gốc."""
        return (e2 for e1 in self.edits1(word) for e2 in self.edits1(e1))
        
    def correct_query(self, query):
        """
        Sửa lỗi nguyên một câu dài.
        """
        words = re.findall(r'\w+', str(query).lower())
        if not words:
            return query
            
        corrected_words = [self.correction(w) for w in words]
        return " ".join(corrected_words)
