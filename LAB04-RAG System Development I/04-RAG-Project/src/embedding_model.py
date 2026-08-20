from sentence_transformers import SentenceTransformer
import config

class EmbeddingModel:
    def __init__(self, model_name=config.EMBEDDING_MODEL_NAME):
        print(f"[embedding] Loading {model_name}...")
        self.model = SentenceTransformer(model_name)

    def encode(self, texts):
        return self.model.encode(
            texts,
            show_progress_bar=True,
            normalize_embeddings=True,
        )

    def encode_query(self, query):   # Convert a single query into an embedding vector
        return self.model.encode([query], normalize_embeddings=True)[0]