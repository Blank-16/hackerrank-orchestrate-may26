import os
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class DocumentRetriever:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.corpora = {
            "hackerrank": [],
            "claude": [],
            "visa": []
        }
        self.vectorizers = {}
        self.tfidf_matrices = {}
        
        self._load_corpus()
        self._build_index()

    def _load_corpus(self):
        print("Loading corpus...")
        for company in self.corpora.keys():
            company_path = os.path.join(self.data_dir, company)
            if not os.path.exists(company_path):
                continue
                
            for root, _, files in os.walk(company_path):
                for file in files:
                    if file.endswith(('.md', '.txt', '.html')):
                        file_path = os.path.join(root, file)
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                                if content.strip():
                                    self.corpora[company].append({
                                        'path': file_path,
                                        'content': content
                                    })
                        except Exception as e:
                            print(f"Error reading {file_path}: {e}")
            print(f"Loaded {len(self.corpora[company])} documents for {company}")

    def _build_index(self):
        print("Building TF-IDF indices...")
        for company, docs in self.corpora.items():
            if not docs:
                continue
            
            texts = [doc['content'] for doc in docs]
            vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
            tfidf_matrix = vectorizer.fit_transform(texts)
            
            self.vectorizers[company] = vectorizer
            self.tfidf_matrices[company] = tfidf_matrix

    def retrieve(self, query, company, top_k=3):
        company = company.lower()
        if company not in self.vectorizers:
            # Fallback to searching all if company is unknown
            all_docs = []
            for c in self.corpora:
                all_docs.extend(self.retrieve(query, c, top_k=1))
            return all_docs[:top_k]

        vectorizer = self.vectorizers[company]
        tfidf_matrix = self.tfidf_matrices[company]
        docs = self.corpora[company]

        if not docs:
            return []

        query_vec = vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
        
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            if similarities[idx] > 0.05: # Minimum similarity threshold
                results.append(docs[idx])
                
        return results
