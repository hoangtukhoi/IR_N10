import math
import time
import pandas as pd
import os

class Evaluation:
    def __init__(self, df=None, excel_path='data/test_query.csv'):
        # Sample ground truth data as fallback
        self.ground_truth = {
            "avatar": {19995},
            "the dark knight": {155},
            "inception": {27205},
            "interstellar": {157336},
            "titanic": {597},
            "the avengers": {24428},
            "iron man": {1726}
        }
        
        self.excel_path = excel_path
        self.load_from_csv(df)

    def load_from_csv(self, df):
        if not os.path.exists(self.excel_path):
            print(f"Warning: {self.excel_path} not found. Using default ground truth.")
            return

        try:
            # Build title to ID mapping
            title_to_id = {}
            if df is not None:
                for _, row in df.iterrows():
                    name = str(row.get('name', row.get('title', ''))).lower().strip()
                    title_to_id[name] = row['id']
            
            loaded_ground_truth = {}
            
            with open(self.excel_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            if len(lines) > 1:
                for line in lines[1:]:  # Skip header
                    line = line.strip()
                    if not line:
                        continue
                        
                    parts = line.split(',')
                    if len(parts) >= 4:
                        # Extract query and expected_title (which might contain commas)
                        q = parts[2].lower().strip()
                        expected_raw = ','.join(parts[3:]).strip()
                        expected_title = expected_raw.lower()
                        
                        ids = set()
                        if expected_title in title_to_id:
                            ids.add(title_to_id[expected_title])
                        elif "the empire strikes back" in expected_title:
                            # Edge case for Star Wars: Episode V
                            if "the empire strikes back" in title_to_id:
                                ids.add(title_to_id["the empire strikes back"])
                                
                        if ids:
                            loaded_ground_truth[q] = ids
            
            if loaded_ground_truth:
                self.ground_truth = loaded_ground_truth
                print(f"Successfully loaded {len(self.ground_truth)} queries from {self.excel_path}")
            else:
                print(f"Warning: CSV loaded but no matching movies found. Using default.")
                
        except Exception as e:
            print(f"Error loading {self.excel_path}: {e}")

    def calculate_metrics(self, query, retrieved_ids, k=10):
        """
        Calculate IR metrics for a single query.
        retrieved_ids: list of IDs returned by the system in order of relevance.
        """
        query = query.lower().strip()
        relevant_ids = self.ground_truth.get(query, set())
        
        if not relevant_ids:
            return None # No ground truth for this query

        retrieved_k = retrieved_ids[:k]
        
        # 1. Precision@K
        tp = len([idx for idx in retrieved_k if idx in relevant_ids])
        precision_at_k = tp / k if k > 0 else 0.0
        
        # 2. Recall@K
        recall_at_k = tp / len(relevant_ids) if len(relevant_ids) > 0 else 0.0
        
        # 3. MRR (Mean Reciprocal Rank) - for this single query it's just Reciprocal Rank
        rr = 0.0
        for i, idx in enumerate(retrieved_ids):
            if idx in relevant_ids:
                rr = 1.0 / (i + 1)
                break
        
        # 4. NDCG@K
        dcg = 0.0
        for i, idx in enumerate(retrieved_k):
            if idx in relevant_ids:
                dcg += 1.0 / math.log(i + 2, 2)
        
        idcg = 0.0
        for i in range(min(len(relevant_ids), k)):
            idcg += 1.0 / math.log(i + 2, 2)
        
        ndcg_at_k = dcg / idcg if idcg > 0 else 0.0
        
        return {
            "p_at_k": precision_at_k,
            "r_at_k": recall_at_k,
            "mrr": rr,
            "ndcg_at_k": ndcg_at_k,
            "tp": tp,
            "relevant_count": len(relevant_ids)
        }

    def calculate_batch_metrics(self, algo_model, df, k=10, prf_n=3, prf_k=3, use_prf=False):
        """
        Runs evaluation on all queries in the ground truth and returns average metrics.
        """
        results = []
        for query, relevant_ids in self.ground_truth.items():
            try:
                if use_prf and hasattr(algo_model, 'get_scores_with_prf'):
                    scores = algo_model.get_scores_with_prf(query, prf_n, prf_k)
                else:
                    scores = algo_model.get_scores(query)
                    
                indices = [i for i, s in sorted(enumerate(scores), key=lambda x: x[1], reverse=True) if s > 0][:k]
                retrieved_ids = df.iloc[indices]['id'].tolist()
                
                metrics = self.calculate_metrics(query, retrieved_ids, k)
                if metrics:
                    results.append(metrics)
            except Exception as e:
                print(f"Error evaluating query '{query}': {e}")
        
        if not results:
            return None
            
        count = len(results)
        avg_metrics = {
            "mPrecision": sum(m['p_at_k'] for m in results) / count,
            "mRecall": sum(m['r_at_k'] for m in results) / count,
            "mRR": sum(m['mrr'] for m in results) / count,
            "mNDCG": sum(m['ndcg_at_k'] for m in results) / count,
            "query_count": count
        }
        return avg_metrics

    def print_batch_report(self, algo_name, avg_metrics):
        """Prints the average metrics for a batch test."""
        if not avg_metrics:
            print(f"\nNo evaluation results for {algo_name}.")
            return
            
        print("\n" + "#"*50)
        print(f"BATCH EVALUATION RESULTS: {algo_name}")
        print(f"Queries evaluated: {avg_metrics['query_count']}")
        print("-" * 50)
        print(f" - Mean Precision@K: {avg_metrics['mPrecision']:.4f}")
        print(f" - Mean Recall@K:    {avg_metrics['mRecall']:.4f}")
        print(f" - Mean Reciprocal Rank: {avg_metrics['mRR']:.4f}")
        print(f" - Mean NDCG@K:      {avg_metrics['mNDCG']:.4f}")
        print("#"*50 + "\n")

    def print_report(self, algo_name, query, retrieved_ids, execution_time, k=10):
        """Prints a formatted report to the terminal."""
        metrics = self.calculate_metrics(query, retrieved_ids, k)
        
        print("\n" + "="*50)
        print(f"ALGORITHM PERFORMANCE REPORT")
        print(f"Algorithm: {algo_name}")
        print(f"Query: '{query}'")
        print(f"Execution Time: {execution_time:.4f} seconds")
        print("-" * 50)
        
        if metrics:
            print(f"Evaluation Metrics (@K={k}):")
            print(f" - Precision@{k}: {metrics['p_at_k']:.4f}")
            print(f" - Recall@{k}:    {metrics['r_at_k']:.4f}")
            print(f" - NDCG@{k}:      {metrics['ndcg_at_k']:.4f}")
            print(f" - RR (Rank):      {metrics['mrr']:.4f}")
            print(f" - Relevant found: {metrics['tp']}/{metrics['relevant_count']}")
        else:
            print("No ground truth available for this query to calculate P/R/NDCG.")
            print(f"Total results retrieved: {len(retrieved_ids)}")
        
        print("="*50 + "\n")
