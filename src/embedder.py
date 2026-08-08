from sentence_transformers import SentenceTransformer
import numpy as np


class Embedder:

    def __init__(
        self,
        model_name="BAAI/bge-base-en-v1.5"
    ):
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        print("Embedding model loaded.")

    def encode_text(self, text):
        """
        Encode a single text.
        """
        embedding = self.model.encode(
            text,
            normalize_embeddings=True,
            convert_to_numpy=True
        )

        return embedding.astype(np.float32)

    def encode_chunks(
        self,
        chunks,
        batch_size=32
    ):
        """
        Encode a list of chunk dictionaries.
        """

        texts = [c["text"] for c in chunks]

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True
        )

        return embeddings.astype(np.float32)

    def add_embeddings(
        self,
        chunks,
        batch_size=32
    ):
        """
        Attach embeddings to chunk dictionaries.
        """

        embeddings = self.encode_chunks(
            chunks,
            batch_size=batch_size
        )

        for chunk, emb in zip(chunks, embeddings):
            chunk["embedding"] = emb

        return chunks