# Movie Retrieval Engine (IR_N10)

A high-performance Movie Retrieval Engine built with Streamlit, implementing hybrid search algorithms (BM25 + TF-IDF) and a modern, vibrant UI.

## 🌟 Features
- **Hybrid Search**: Combines BM25 and TF-IDF for more accurate information retrieval.
- **Pseudo-Relevance Feedback (PRF)**: Enhances search results based on the top relevant documents.
- **Modern UI**: Responsive design with Aurora gradients and Glassmorphism aesthetics.
- **Smart Filtering**: Filter movies by Genre, Year, and Rating.
- **Interactive Details**: View full movie information including cast, director, budget, and revenue.
- **Spell Checker**: Suggests corrections for misspelled search queries.

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- [Pandas](https://pandas.pydata.org/)
- [Streamlit](https://streamlit.io/)
- [NLTK](https://www.nltk.org/)
- [Scikit-learn](https://scikit-learn.org/)

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/hoangtukhoi/IR_N10.git
   cd IR_N10
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *(Note: Ensure you have the TMDB 5000 dataset files in the `data/` directory)*

### Running the App
```bash
streamlit run app.py
```

## 🛠 Technology Stack
- **Frontend**: Streamlit, Custom CSS (Glassmorphism)
- **Algorithms**: BM25, TF-IDF, Hybrid Search, PRF
- **Data Handling**: Pandas, JSON
- **Natural Language Processing**: NLTK (Tokenization, Preprocessing)

## 📁 Repository Structure
- `app.py`: Main Streamlit application entry point.
- `algorithm/`: Implementation of search algorithms (BM25, TF-IDF, Hybrid).
- `frontend/`: UI components and custom styling.
- `data/`: Dataset storage (CSV files).

## 👥 Contributors
- **IR_N10 Team**

---
*Built with ❤️ for the Information Retrieval course.*
