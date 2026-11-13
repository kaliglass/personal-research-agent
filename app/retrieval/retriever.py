from app.retrieval.embeddings import BGEEmbedder
from app.retrieval.vector_store import FaissVectorStore
from app.schemas.retrieval import SearchResult


class Retriever:
    """Coordinate query embedding and FAISS similarity search."""

    def __init__(
        self,
        embedder: BGEEmbedder,
        vector_store: FaissVectorStore,
    ) -> None:
        # Ensure both components use the same vector dimensions.
        if embedder.embedding_dimension != vector_store.embedding_dimension:
            raise ValueError("Embedder and vector store dimensions do not match")

        self.embedder = embedder
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Retrieve the most similar document chunks for a query."""

        if not isinstance(query, str):
            raise TypeError("Query must be a string")

        if not query.strip():
            raise ValueError("Query cannot be empty")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        # Step 1: Convert the question into an embedding.
        query_embedding = self.embedder.embed_query(query.strip())

        # Step 2: Search the FAISS index.
        results = self.vector_store.search(
            query_embedding,
            top_k=top_k,
        )

        return results
