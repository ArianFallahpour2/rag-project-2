import faiss
import numpy as np
import requests
import streamlit as st

# Load the FAISS index (Extracted from data.zip)
index = faiss.read_index("child_index.faiss")

# Get Hugging Face API key from Streamlit secrets
HF_TOKEN = st.secrets.get("HF_TOKEN")
API_URL = "https://router.huggingface.co/v1/models/BAAI/bge-m3"


def get_query_embedding_api(query_text):
  """Sends the query to Hugging Face Serverless API and gets back a 1024-dim vector."""
  headers = {"Authorization": f"Bearer {HF_TOKEN}"} if HF_TOKEN else {}
  payload = {"inputs": [query_text], "options": {"wait_for_model": True}}

  response = requests.post(API_URL, headers=headers, json=payload)

  if response.status_code != 200:
    raise RuntimeError(
        f"HF Embedding API failed [{response.status_code}]: {response.text}"
    )

  # HF Feature Extraction API returns a list of embeddings
  data = response.json()

  # Extract embedding vector and cast to float32 for FAISS
  embedding = np.array(data[0], dtype="float32")

  # Normalize for cosine similarity / inner product matching
  faiss.normalize_L2(embedding)

  return embedding


def child_retriever(input_query, input_child_chunks):
  """Retrieve child chunk texts directly for a given query.

  Steps:
    1. Embed query via Hugging Face Serverless API (Zero local RAM used!).
    2. Search FAISS index for top 5 matches.
    3. Return matching chunk texts.
  """
  # Get 1024-dim query vector via API
  embedded_query = get_query_embedding_api(input_query)

  # FAISS expects 2D array (shape: 1 x 1024)
  if len(embedded_query.shape) == 1:
    embedded_query = np.expand_dims(embedded_query, axis=0)

  # Get top-5 child indices
  distances, indices = index.search(embedded_query, 5)
  top_child_indices = indices[0]

  # Return actual texts from child chunks
  retrieved_texts = [input_child_chunks[i]["text"] for i in top_child_indices]
  return retrieved_texts
