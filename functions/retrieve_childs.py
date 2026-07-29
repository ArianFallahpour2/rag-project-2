import os
import time
import faiss
import numpy as np
import requests

# Retrieve HF_TOKEN from ParsPack Environment Variables
HF_TOKEN = os.getenv("HF_TOKEN")


def get_query_embedding_api(input_query):
    API_URL = "https://router.huggingface.co/hf-inference/models/BAAI/bge-m3/pipeline/feature-extraction"
    
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }
    
    payload = {"inputs": input_query}

    for attempt in range(3):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=20)
            
            if response.status_code == 200:
                return response.json()
            
            if response.status_code == 503:
                print("Embedding model is loading... retrying in 10s.")
                time.sleep(10)
                continue
                
            print(f"HF API Error [{response.status_code}]: {response.text}")
            return None

        except requests.exceptions.RequestException as e:
            print(f"Network error calling embedding API: {e}")
            return None

    return None


def child_retriever(input_query, input_child_chunks, faiss_index_path="child_index.faiss"):
    """Retrieve child chunk texts directly for a given query."""
    
    # Load index dynamically to safely handle startup extraction
    if not os.path.exists(faiss_index_path):
        raise FileNotFoundError(f"FAISS index file not found at: {faiss_index_path}")
        
    index = faiss.read_index(faiss_index_path)

    raw_embedding = get_query_embedding_api(input_query)
    if raw_embedding is None:
        return ["Error: Failed to fetch query embedding from Hugging Face API."]

    # Convert list to float32 NumPy array
    embedded_query = np.array(raw_embedding, dtype=np.float32)

    # Dimensionality check & reduction if sequence dim present
    if embedded_query.ndim > 1:
        embedded_query = np.mean(embedded_query, axis=tuple(range(embedded_query.ndim - 1)))

    if embedded_query.ndim == 1:
        embedded_query = np.expand_dims(embedded_query, axis=0)

    # Query FAISS index
    distances, indices = index.search(embedded_query, 5)
    top_child_indices = indices[0]

    retrieved_texts = [
        input_child_chunks[i]["text"] 
        for i in top_child_indices 
        if 0 <= i < len(input_child_chunks)
    ]
    
    return retrieved_texts
