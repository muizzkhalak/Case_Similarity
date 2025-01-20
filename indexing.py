import faiss
import torch
from typing import List, Dict, Any
from collections import defaultdict

class ApproxNN:

    def __init__(self, 
                 embeddings, 
                 mapping_dict):

        if isinstance(embeddings, torch.Tensor):
            embeddings = embeddings.cpu().numpy()

        self.mapping_dict = mapping_dict
        self.dimension = embeddings.shape[1]  # Dimensionality of the embeddings
        self.embeddings = embeddings.astype('float32') # FAISS expects embeddings to be in float32 format

        self.create_index()

    def create_index(self):

        self.index = faiss.IndexFlatIP(self.dimension)  # Using a flat (brute-force) index

        faiss.normalize_L2(self.embeddings) # Normalize Embeddings

        self.index.add(self.embeddings)

    def query(self, query_id, top_k):

        # Query for the first sentence embedding
        qidx = self.mapping_dict[query_id]
        query_embedding = self.embeddings[qidx].reshape(1,self.dimension)
        faiss.normalize_L2(query_embedding) # Normalize Embedding

        # Searching the index
        k = top_k + 1  # Number of nearest neighbors
        distances, indices = self.index.search(query_embedding, k)

        results = [list(self.mapping_dict.keys())[list(self.mapping_dict.values()).index(idx)] for idx in indices[0][1:]]
        return results


def weighted_reciprocal_rank_fusion(
    ranked_lists: List[List[Any]],
    weights: List[float],
    k: int = 60
) -> List[Any]:
    """
    Perform Weighted Reciprocal Rank Fusion on multiple ranked lists.

    Args:
        ranked_lists (List[List[Any]]): A list of ranked lists, where each ranked list is a list of document identifiers.
        weights (List[float]): A list of weights corresponding to each ranked list.
        k (int, optional): The constant k used in the RRF formula. Defaults to 60.

    Returns:
        List[Any]: A single fused ranked list of document identifiers.
    """
    if len(ranked_lists) != len(weights):
        raise ValueError("The number of ranked lists must match the number of weights.")

    # Initialize a dictionary to hold the scores for each document
    scores = defaultdict(float)

    for idx, (ranked_list, weight) in enumerate(zip(ranked_lists, weights)):
        # print(f"Processing ranker {idx + 1} with weight {weight}")
        for rank, doc in enumerate(ranked_list, start=1):
            contribution = weight / (k + rank)
            scores[doc] += contribution
            # print(f"  Document: {doc}, Rank: {rank}, Contribution: {contribution:.6f}, Total Score: {scores[doc]:.6f}")

    # Sort the documents based on their total scores in descending order
    fused_ranking = sorted(scores.items(), key=lambda item: item[1], reverse=True)

    # Extract and return the document identifiers from the sorted list
    return [doc for doc, score in fused_ranking]