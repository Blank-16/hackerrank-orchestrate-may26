import os
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity

class Chunk:
    def __init__(self, chunk_id, parent_id, text, company):
        self.chunk_id = chunk_id
        self.parent_id = parent_id
        self.text = text
        self.company = company
        self.token_count = len(text.split())

class AdvancedRetriever:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.chunks = []
        
        print("Loading SentenceTransformer models (this might take a moment on first run)...")
        # Load local lightweight dense model and reranker
        self.dense_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        
        self.bm25_index = None
        self.dense_embeddings = None
        
        self._load_and_chunk_corpus()
        self._build_indices()

    def _load_and_chunk_corpus(self):
        print("Loading and chunking corpus...")
        chunk_size = 800  # Characters
        overlap = 150
        
        for company in ["hackerrank", "claude", "visa"]:
            company_path = os.path.join(self.data_dir, company)
            if not os.path.exists(company_path):
                continue
                
            for root, _, files in os.walk(company_path):
                for file in files:
                    if file.endswith(('.md', '.txt', '.html')):
                        file_path = os.path.join(root, file)
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read().strip()
                                if not content:
                                    continue
                                
                                # Simple character chunking
                                start = 0
                                chunk_idx = 0
                                parent_id = f"{company}_{file}"
                                
                                while start < len(content):
                                    end = min(start + chunk_size, len(content))
                                    chunk_text = content[start:end]
                                    chunk_id = f"{parent_id}_chunk_{chunk_idx}"
                                    
                                    self.chunks.append(Chunk(
                                        chunk_id=chunk_id,
                                        parent_id=parent_id,
                                        text=chunk_text,
                                        company=company
                                    ))
                                    
                                    start += (chunk_size - overlap)
                                    chunk_idx += 1
                        except Exception as e:
                            pass
        print(f"Loaded {len(self.chunks)} chunks across all companies.")

    def _build_indices(self):
        print("Building BM25 and Dense indices...")
        if not self.chunks:
            return
            
        # 1. BM25 Sparse Index
        tokenized_corpus = [chunk.text.lower().split() for chunk in self.chunks]
        self.bm25_index = BM25Okapi(tokenized_corpus)
        
        # 2. Dense Vector Index
        corpus_texts = [chunk.text for chunk in self.chunks]
        self.dense_embeddings = self.dense_model.encode(corpus_texts, convert_to_tensor=False)
        self.dense_embeddings = np.array(self.dense_embeddings)

    def retrieve(self, query: str, target_company: str, top_k=10):
        if not self.chunks:
            return [], "no_docs"
            
        # 1. Sparse Search
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25_index.get_scores(tokenized_query)
        
        # 2. Dense Search
        query_embedding = self.dense_model.encode([query])[0]
        dense_scores = cosine_similarity([query_embedding], self.dense_embeddings)[0]
        
        # Filter chunks by company before RRF
        valid_indices = [i for i, c in enumerate(self.chunks) if target_company == "unknown" or c.company == target_company.lower()]
        
        if not valid_indices:
            # Fallback to all companies
            valid_indices = list(range(len(self.chunks)))
            
        sparse_ranked = sorted(valid_indices, key=lambda i: bm25_scores[i], reverse=True)
        dense_ranked = sorted(valid_indices, key=lambda i: dense_scores[i], reverse=True)
        
        # 3. Pure Rank-Based RRF Fusion
        fused_scores = {}
        k_rrf = 60
        
        for rank, idx in enumerate(sparse_ranked):
            fused_scores[idx] = fused_scores.get(idx, 0.0) + 1.0 / (k_rrf + rank + 1)
            
        for rank, idx in enumerate(dense_ranked):
            fused_scores[idx] = fused_scores.get(idx, 0.0) + 1.0 / (k_rrf + rank + 1)
            
        # Sort by fused score
        sorted_fused_indices = sorted(fused_scores.keys(), key=lambda idx: fused_scores[idx], reverse=True)
        
        # Top 10 documents passed to Cross-Encoder
        docs_to_score = [self.chunks[idx] for idx in sorted_fused_indices[:top_k]]
        
        # 4. Cross Encoder Reranking
        pairs = [(query, doc.text) for doc in docs_to_score]
        reranker_scores = self.cross_encoder.predict(pairs)
        
        scored_docs = list(zip(docs_to_score, reranker_scores))
        max_score = max(reranker_scores) if len(reranker_scores) > 0 else 0
        
        # 5. Hard Confidence Gate
        MIN_SCORE_THRESHOLD = 0.45
        if max_score < MIN_SCORE_THRESHOLD:
            return [], "low_confidence"
            
        # 6. Relative Thresholding & Diversity Filtering
        relative_threshold = 0.75 * max_score
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        selected_context = []
        seen_chunk_ids = set()
        source_doc_counts = {}
        current_tokens = 0
        max_tokens = 2000
        max_chunks_per_doc = 2
        
        for doc, score in scored_docs:
            if score < relative_threshold:
                continue
            if doc.chunk_id in seen_chunk_ids:
                continue
            if source_doc_counts.get(doc.parent_id, 0) >= max_chunks_per_doc:
                continue
            if current_tokens + doc.token_count > max_tokens:
                continue
                
            selected_context.append(doc)
            seen_chunk_ids.add(doc.chunk_id)
            source_doc_counts[doc.parent_id] = source_doc_counts.get(doc.parent_id, 0) + 1
            current_tokens += doc.token_count
            
        return selected_context, "pass"
