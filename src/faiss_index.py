import pickle
from pathlib import Path

import faiss
import numpy as np


class FaissIndex:
    """
    FAISS vector index for semantic retrieval.

    Design:
        • Exact Nearest Neighbor Search
        • Cosine Similarity using IndexFlatIP
        • L2-normalized float32 embeddings
        • Chunk metadata stored separately

    Suitable for:
        • Legal RAG
        • Contract Retrieval
        • Semantic Search
    """

    def __init__(self):

        self.index = None
        self.chunks = []

    ####################################################################
    # Build Index
    ####################################################################

    def build_index(
        self,
        embeddings,
        chunks
    ):
        """
        Build a FAISS index from chunk embeddings.

        Parameters
        ----------
        embeddings : ndarray (N, D)
            Float32 normalized embeddings.

        chunks : list
            Chunk dictionaries corresponding
            to the embeddings.
        """

        embeddings = np.asarray(
            embeddings,
            dtype=np.float32
        )

        if embeddings.ndim != 2:
            raise ValueError(
                "Embeddings must be a 2D array."
            )

        if len(embeddings) != len(chunks):
            raise ValueError(
                "Number of embeddings and chunks must match."
            )

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(dimension)

        self.index.add(embeddings)

        self.chunks = list(chunks)

        print(f"Indexed {len(self.chunks)} chunks.")

    ####################################################################
    # Search
    ####################################################################

    def search(
        self,
        query_embedding,
        top_k=5
    ):
        """
        Retrieve the most similar chunks.

        Parameters
        ----------
        query_embedding : ndarray
            Normalized query embedding.

        top_k : int
            Number of results.

        Returns
        -------
        list
            Retrieved chunk dictionaries
            with similarity scores.
        """

        if self.index is None:
            raise RuntimeError(
                "Index has not been built."
            )

        top_k = min(
            top_k,
            len(self.chunks)
        )

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32
        ).reshape(1, -1)

        scores, indices = self.index.search(
            query_embedding,
            top_k
        )

        results = []

        for score, idx in zip(
            scores[0],
            indices[0]
        ):

            if idx == -1:
                continue

            chunk = self.chunks[idx].copy()

            chunk["score"] = float(score)

            results.append(chunk)

        return results

    ####################################################################
    # Save Index
    ####################################################################

    def save(
        self,
        index_path,
        metadata_path
    ):
        """
        Save FAISS index and chunk metadata.
        """

        if self.index is None:
            raise RuntimeError(
                "No FAISS index available."
            )

        index_path = Path(index_path)
        metadata_path = Path(metadata_path)

        index_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        metadata_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        faiss.write_index(
            self.index,
            str(index_path)
        )

        with open(
            metadata_path,
            "wb"
        ) as f:

            pickle.dump(
                self.chunks,
                f
            )

        print("FAISS index saved.")

    ####################################################################
    # Load Index
    ####################################################################

    def load(
        self,
        index_path,
        metadata_path
    ):
        """
        Load FAISS index and metadata.
        """

        index_path = Path(index_path)
        metadata_path = Path(metadata_path)

        if not index_path.exists():
            raise FileNotFoundError(index_path)

        if not metadata_path.exists():
            raise FileNotFoundError(metadata_path)

        self.index = faiss.read_index(
            str(index_path)
        )

        with open(
            metadata_path,
            "rb"
        ) as f:

            self.chunks = pickle.load(f)

        print(
            f"Loaded {len(self.chunks)} chunks."
        )

    ####################################################################
    # Information
    ####################################################################

    @property
    def size(self):
        """
        Number of indexed vectors.
        """

        if self.index is None:
            return 0

        return self.index.ntotal

    @property
    def dimension(self):
        """
        Embedding dimension.
        """

        if self.index is None:
            return None

        return self.index.d