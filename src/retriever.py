import os
import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class Retriever:
    def __init__(self, index_path, metadata_path, embedding_model_name="all-MiniLM-L6-v2"):
        """
        Args:
            index_path: Path to faiss_index.bin
            metadata_path: Path to chunks_metadata.json
            embedding_model_name: The HuggingFace model used to embed the chunks.
        """
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.embedding_model_name = embedding_model_name
        
        self._load_resources()

    def _load_resources(self):
        # 1. Load Embedding Model
        print(f"Loading embedding model: {self.embedding_model_name}...")
        self.embedder = SentenceTransformer(self.embedding_model_name)
        
        # 2. Load FAISS Index
        print(f"Loading FAISS index from {self.index_path}...")
        if not os.path.exists(self.index_path):
            raise FileNotFoundError(f"FAISS index not found at {self.index_path}")
        self.index = faiss.read_index(self.index_path)
        
        # 3. Load Metadata (Robust Fix for JSON/JSONL)
        print(f"Loading metadata from {self.metadata_path}...")
        if not os.path.exists(self.metadata_path):
             raise FileNotFoundError(f"Metadata not found at {self.metadata_path}")

        with open(self.metadata_path, 'r', encoding='utf-8') as f:
            try:
                # Attempt to load as standard JSON (List or Dict)
                data = json.load(f)
                
                # If it's a list, convert to dict with string keys to match FAISS indices
                if isinstance(data, list):
                    self.metadata = {str(i): doc for i, doc in enumerate(data)}
                else:
                    self.metadata = data
                    
            except json.JSONDecodeError:
                # Fallback: Load as JSONL (one object per line)
                print("Standard JSON load failed. Trying JSONL format...")
                f.seek(0)
                self.metadata = {}
                for i, line in enumerate(f):
                    if line.strip():
                        try:
                            self.metadata[str(i)] = json.loads(line)
                        except json.JSONDecodeError as e:
                            print(f"Skipping malformed line {i}: {e}")

        print(f"Successfully loaded {len(self.metadata)} documents.")

    def search(self, query, top_k=3):
        # 1. Convert query to vector
        query_vec = self.embedder.encode([query])
        query_vec = np.array(query_vec).astype(np.float32)

        # 2. Dimension Check & Fix
        if self.index.d != query_vec.shape[1]:
            # Reshape if it's 1D
            if len(query_vec.shape) == 1:
                 query_vec = query_vec.reshape(1, -1)
            
            # If still mismatch, raise clear error
            if self.index.d != query_vec.shape[1]:
                raise ValueError(
                    f"Dimension mismatch! Index expects {self.index.d}, "
                    f"but embedding model produced {query_vec.shape[1]}. "
                    "Change 'embedding_model_name' in 'src/rag_pipeline.py' to match your index."
                )

        # 3. Search
        scores, indices = self.index.search(query_vec, top_k)
        
        # 4. Retrieve Results
        results = []
        for i, idx in enumerate(indices[0]):
            if idx != -1:  # -1 means no match found
                idx_key = str(idx) 
                if idx_key in self.metadata:
                    doc = self.metadata[idx_key]
                    results.append({
                        "content": doc.get("content", "") or doc.get("text", ""), # Handle 'content' or 'text' keys
                        "source": doc.get("source", "Unknown"),
                        "score": float(scores[0][i])
                    })
        
        return results
