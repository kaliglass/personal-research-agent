from collections.abc import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer

from app.schemas.chunk import DocumentChunk

BGE_MODEL_NAME = "BAAI/bge-base-en-v1.5"
BGE_EMBEDDING_DIMENSION = 768
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


class BGEEmbedder:
    """Generate normalized embeddings using the BGE model."""

    def __init__(self, device: str | None = None) -> None:
        # Load the model once and reuse it for every embedding operation.
        self.model = SentenceTransformer(
            BGE_MODEL_NAME,
            device=device,
        )

        self.embedding_dimension = self.model.get_sentence_embedding_dimension()

        if self.embedding_dimension != BGE_EMBEDDING_DIMENSION:
            raise RuntimeError(
                f"Expected {BGE_EMBEDDING_DIMENSION} dimensions, "
                f"got {self.embedding_dimension}"
            )

        self.device = str(self.model.device)

    def embed_texts(
        self,
        texts: Sequence[str],
        batch_size: int = 16,
    ) -> np.ndarray:
        """Convert a collection of texts into normalized vectors."""

        if isinstance(texts, str):
            raise TypeError("Expected a sequence of texts, not one string")

        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")

        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Texts must be non-empty strings")

        if not texts:
            return np.empty(
                (0, self.embedding_dimension),
                dtype=np.float32,
            )

        embeddings = self.model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        embeddings = np.asarray(embeddings, dtype=np.float32)

        expected_shape = (len(texts), self.embedding_dimension)

        if embeddings.shape != expected_shape:
            raise RuntimeError(
                f"Unexpected embedding shape: {embeddings.shape}; "
                f"expected {expected_shape}"
            )

        if not np.isfinite(embeddings).all():
            raise RuntimeError("Embeddings contain invalid numerical values")

        return embeddings

    def embed_chunks(
        self,
        chunks: Sequence[DocumentChunk],
        batch_size: int = 16,
    ) -> np.ndarray:
        """Generate one embedding for each document chunk."""

        texts = [chunk.text for chunk in chunks]

        return self.embed_texts(texts, batch_size=batch_size)

    def embed_query(self, query: str) -> np.ndarray:
        """Generate an embedding for a retrieval query."""

        if not isinstance(query, str) or not query.strip():
            raise ValueError("Query must be a non-empty string")

        instructed_query = BGE_QUERY_INSTRUCTION + query.strip()

        return self.embed_texts([instructed_query])
