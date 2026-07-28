import faiss
import numpy as np
import requests
import streamlit as st
import time

# Load the FAISS index (Extracted from data.zip)
index = faiss.read_index("child_index.faiss")

# Get Hugging Face API key from Streamlit secrets
HF_TOKEN = st.secrets.get("HF_TOKEN")


def get_query_embedding_api(input_query):
    # Correct serverless feature-extraction endpoint
    API_URL = "https://router.huggingface.co/hf-inference/models/BAAI/bge-m3/pipeline/feature-extraction"
    
    headers = {
        "Authorization": f"Bearer {HF_TOKEN}",
        "Content-Type": "application/json"
    }
    
    payload = {"inputs": input_query}

    # Retry up to 3 times if the model is waking up (cold start)
    for attempt in range(3):
        try:
            response = requests.post(API_URL, headers=headers, json=payload, timeout=20)
            
            # Successful response
            if response.status_code == 200:
                return response.json()
            
            # Model is loading into memory - wait and retry
            if response.status_code == 503:
                st.warning("Embedding model is waking up... retrying in 10 seconds.")
                time.sleep(10)
                continue
                
            # Other errors
            st.error(f"HF API returned status {response.status_code}: {response.text}")
            return None

        except requests.exceptions.RequestException as e:
            st.error(f"Network error calling embedding API: {e}")
            return None

    return None


def child_retriever(input_query, input_child_chunks):
    """Retrieve child chunk texts directly for a given query.

    Steps:
      1. Embed query via Hugging Face Serverless API.
      2. Convert JSON response to NumPy float32 array.
      3. Search FAISS index for top 5 matches.
      4. Return matching chunk texts.
    """
    # 1. Get embedding via API
    raw_embedding = get_query_embedding_api(input_query)

    # Safety check: Handle API connection failures
    if raw_embedding is None:
        st.error("Could not retrieve embedding from API.")
        return []

    # 2. Convert Python list to float32 NumPy array
    embedded_query = np.array(raw_embedding, dtype=np.float32)

    # Flatten if HF returned an extra dimension (e.g., shape [1, 1024] or [1, seq_len, 1024])
    if embedded_query.ndim > 1:
        embedded_query = np.mean(embedded_query, axis=tuple(range(embedded_query.ndim - 1)))

    # Ensure shape is 2D for FAISS (shape: 1 x 1024)
    if embedded_query.ndim == 1:
        embedded_query = np.expand_dims(embedded_query, axis=0)

    # 3. Get top-5 child indices from FAISS
    distances, indices = index.search(embedded_query, 5)
    top_child_indices = indices[0]

    # 4. Return actual texts from child chunks
    retrieved_texts = [
        input_child_chunks[i]["text"] 
        for i in top_child_indices 
        if 0 <= i < len(input_child_chunks)
    ]
    
    return retrieved_texts
