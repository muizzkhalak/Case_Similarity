from sentence_transformers import models, SentenceTransformer
    
class SentenceBERT:

    def __init__(self, model_name, pooling, device):

        transformer = models.Transformer(model_name)
        pooling = models.Pooling(transformer.get_word_embedding_dimension(), pooling_mode=pooling)
        self.model = SentenceTransformer(modules=[transformer, pooling], device=device)

    def encode(self, texts):

        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(list(texts), show_progress_bar=True)

        return embeddings
    
