import numpy as np
import pytest

from app.retrieval.retriever import Retriever
from app.schemas.chunk import DocumentChunk
from app.schemas.retrieval import SearchResult


class FakeEmbedder:
    """Simulate query embedding generation."""

    embedding_dimension = 3

    def __init__(self):
        self.received_query = None

    def embed_query(self, query: str) -> np.ndarray:
        self.received_query = query

        return np.array(
            [[1.0, 0.0, 0.0]],
            dtype=np.float32,
        )


class FakeVectorStore:
    """Simulate FAISS similarity search."""

    embedding_dimension = 3

    def __init__(self):
        self.received_embedding = None
        self.received_top_k = None

        self.results = [
            SearchResult(
                chunk=DocumentChunk(
                    chunk_id="test.pdf:page-1:chunk-0",
                    document_name="test.pdf",
                    source_path="data/library/test.pdf",
                    page_number=1,
                    chunk_index=0,
                    token_count=5,
                    text="The Transformer uses self-attention.",
                ),
                score=0.95,
            )
        ]

    def search(self, query_embedding, top_k=5):
        self.received_embedding = query_embedding
        self.received_top_k = top_k

        return self.results


@pytest.fixture
def retriever_components():
    embedder = FakeEmbedder()
    vector_store = FakeVectorStore()

    retriever = Retriever(
        embedder=embedder,
        vector_store=vector_store,
    )

    return retriever, embedder, vector_store


def test_retrieve_generates_embedding_and_searches(
    retriever_components,
):
    retriever, embedder, vector_store = retriever_components

    question = "What is self-attention?"

    results = retriever.retrieve(question, top_k=3)

    assert embedder.received_query == question
    assert vector_store.received_embedding.shape == (1, 3)
    assert vector_store.received_top_k == 3

    assert len(results) == 1
    assert results[0].chunk.text == ("The Transformer uses self-attention.")
    assert results[0].score == pytest.approx(0.95)


def test_retrieve_rejects_empty_query(retriever_components):
    retriever, embedder, vector_store = retriever_components

    with pytest.raises(ValueError):
        retriever.retrieve("   ")

    assert embedder.received_query is None
    assert vector_store.received_embedding is None


def test_retrieve_rejects_invalid_top_k(retriever_components):
    retriever, embedder, vector_store = retriever_components

    with pytest.raises(ValueError):
        retriever.retrieve("What is attention?", top_k=0)

    assert embedder.received_query is None
    assert vector_store.received_embedding is None


def test_retriever_rejects_dimension_mismatch():
    embedder = FakeEmbedder()

    vector_store = FakeVectorStore()
    vector_store.embedding_dimension = 768

    with pytest.raises(ValueError):
        Retriever(
            embedder=embedder,
            vector_store=vector_store,
        )
