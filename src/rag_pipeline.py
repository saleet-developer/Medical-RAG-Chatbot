# import os
# from src.retriever import Retriever
# from src.generator import Generator

# # Configuration Paths
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# MODEL_PATH = os.path.join(BASE_DIR, "models", "phi-3-mini-4k-instruct.Q4_K_M.gguf")
# INDEX_PATH = os.path.join(BASE_DIR, "data", "faiss_index.bin")
# METADATA_PATH = os.path.join(BASE_DIR, "data", "chunks_metadata.json")

# class RAGPipeline:
#     def __init__(self):
#         # IMPORTANT: If you get a dimension error, change 'all-MiniLM-L6-v2' 
#         # to the model you used to create your index (e.g., 'all-mpnet-base-v2').
#         self.retriever = Retriever(INDEX_PATH, METADATA_PATH, embedding_model_name="all-MiniLM-L6-v2")
        
#         # Initialize Generator
#         self.generator = Generator(MODEL_PATH)

#     def query(self, question, top_k=3, max_tokens=256, temperature=0.1):
#         print(f"Retrieving context for: {question}")
#         docs = self.retriever.search(question, top_k=top_k)
        
#         # Combine retrieved documents into a single context string
#         context_text = "\n\n".join([f"[Source: {d['source']}]\n{d['content']}" for d in docs])
        
#         if not context_text:
#             context_text = "No relevant medical context found."

#         print("Generating answer...")
#         answer = self.generator.generate(context_text, question, max_tokens, temperature)
        
#         return {
#             "answer": answer,
#             "source_documents": docs
#         }



import os
from src.retriever import Retriever
from src.generator import Generator

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "phi-3-mini-4k-instruct.Q4_K_M.gguf")
INDEX_PATH = os.path.join(BASE_DIR, "data", "faiss_index.bin")
METADATA_PATH = os.path.join(BASE_DIR, "data", "chunks_metadata.json")

class RAGPipeline:
    def __init__(self):
        self.retriever = Retriever(INDEX_PATH, METADATA_PATH, embedding_model_name="all-MiniLM-L6-v2")        
        self.generator = Generator(MODEL_PATH)

    def query(self, question, top_k=3, max_tokens=256, temperature=0.1):
        docs = self.retriever.search(question, top_k=top_k)        
        context_text = "\n\n".join([f"[Source: {d['source']}]\n{d['content']}" for d in docs])       
        
        if not context_text:
            context_text = "No relevant medical context found."
        
        # Returns the generator object directly
        answer_stream = self.generator.generate(context_text, question, max_tokens, temperature)        
        
        return {
            "answer_stream": answer_stream,
            "source_documents": docs
        }

