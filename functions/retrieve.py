# functions/retrieve.py
import faiss
import numpy as np
from functions.retrieve_childs import get_query_embedding_api

index = faiss.read_index("child_index.faiss")

def retriever(input_query, input_child_chunks, input_parent_list):
    # Reuse the imported function!
    embedded_query = get_query_embedding_api(input_query)
    
    distances, indices = index.search(embedded_query, 3)
    top_child_indices = indices[0]

    seen = set()
    unique_parent_ids = []
    for i in top_child_indices:
        parent_id = input_child_chunks[i]["parent_id"]
        if parent_id not in seen:
            seen.add(parent_id)
            unique_parent_ids.append(parent_id)

    return [input_parent_list[pid] for pid in unique_parent_ids]
