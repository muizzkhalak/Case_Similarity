import faiss
import torch

class ApproxNN:

    def __init__(self, embeddings, mapping_dict):

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
