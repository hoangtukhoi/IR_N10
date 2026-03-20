import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)

def clean_text(text):
  text = text.lower()
  text = re.sub(r"http\S+|www\S+", " ", text)
  text = re.sub(r"[^a-zA-ZÀ-ỹ\s]", " ", text)
  text = re.sub(r"\s+", " ", text).strip()
  return text

def tokenize_and_remove_stopwords(text):
    text = clean_text(text)
    tokens = text.split()
    tokens = [w for w in tokens if w not in stopwords]
    return " ".join(tokens)