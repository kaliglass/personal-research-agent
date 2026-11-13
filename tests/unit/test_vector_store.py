import numpy as np
import pytest

from app.retrieval.vector_store import FaissVectorStore
from app.schemas.chunk import DocumentChunk


def make_chunk(index: int) -> DocumentChunk:
    """Create a sample document chunk for testing."""

    return DocumentChunk(
        chunk_id=f"test.pdf:page-1:chunk-{index}",
        document_name="test.pdf",
        source_path="data/library/test.pdf",
        page_number=1,
        chunk_index=index,
        token_count=5,
        text=f"This is test passage {index}.",
    )


@pytest.fixture
def sample_store():
    """Create a FAISS store containing three sample vectors."""

    store = FaissVectorStore(embedding_dimension=3)

    chunks = [
        make_chunk(0),
        make_chunk(1),
        make_chunk(2),
    ]

    vectors = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.6, 0.8, 0.0],
        ],
        dtype=np.float32,
    )

    store.add(chunks, vectors)

    return store


# --------------------------------------------------
# INDEX CREATION AND ADDITION
# --------------------------------------------------


def test_add_vectors_updates_index_size(sample_store):
    assert sample_store.size == 3


# --------------------------------------------------
# VECTOR SEARCH
# --------------------------------------------------


def test_search_returns_results_in_similarity_order(sample_store):
    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    results = sample_store.search(query, top_k=3)

    assert len(results) == 3

    assert results[0].chunk.chunk_index == 0
    assert results[1].chunk.chunk_index == 2
    assert results[2].chunk.chunk_index == 1

    assert results[0].score == pytest.approx(1.0)
    assert results[1].score == pytest.approx(0.6)
    assert results[2].score == pytest.approx(0.0)


def test_search_returns_original_chunk_metadata(sample_store):
    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    results = sample_store.search(query, top_k=1)

    result = results[0]
    chunk = result.chunk

    assert chunk.document_name == "test.pdf"
    assert chunk.page_number == 1
    assert chunk.chunk_index == 0
    assert chunk.chunk_id == "test.pdf:page-1:chunk-0"
    assert chunk.text == "This is test passage 0."


def test_search_respects_top_k(sample_store):
    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    results = sample_store.search(query, top_k=2)

    assert len(results) == 2


def test_search_limits_results_to_available_vectors(sample_store):
    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    results = sample_store.search(query, top_k=10)

    assert len(results) == 3


def test_empty_store_returns_empty_results():
    store = FaissVectorStore(embedding_dimension=3)

    results = store.search(np.array([1.0, 0.0, 0.0], dtype=np.float32))

    assert results == []


# --------------------------------------------------
# INPUT VALIDATION
# --------------------------------------------------


def test_mismatched_chunk_and_vector_counts_are_rejected():
    store = FaissVectorStore(embedding_dimension=3)

    with pytest.raises(ValueError):
        store.add(
            [make_chunk(0), make_chunk(1)],
            np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
        )


def test_wrong_embedding_dimension_is_rejected():
    store = FaissVectorStore(embedding_dimension=3)

    with pytest.raises(ValueError):
        store.add(
            [make_chunk(0)],
            np.array([[1.0, 0.0]], dtype=np.float32),
        )


def test_unnormalized_vectors_are_rejected():
    store = FaissVectorStore(embedding_dimension=3)

    with pytest.raises(ValueError):
        store.add(
            [make_chunk(0)],
            np.array([[2.0, 0.0, 0.0]], dtype=np.float32),
        )


def test_duplicate_chunk_ids_are_rejected():
    store = FaissVectorStore(embedding_dimension=3)

    chunk = make_chunk(0)

    vectors = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )

    with pytest.raises(ValueError):
        store.add([chunk, chunk], vectors)


def test_invalid_top_k_is_rejected(sample_store):
    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    with pytest.raises(ValueError):
        sample_store.search(query, top_k=0)


# --------------------------------------------------
# PERSISTENCE
# --------------------------------------------------


def test_save_and_load_preserves_search_results(sample_store, tmp_path):
    # Search before saving.
    query = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    original_results = sample_store.search(query, top_k=2)

    # Save the index and metadata.
    sample_store.save(tmp_path)

    # Load a new vector store from disk.
    loaded_store = FaissVectorStore.load(tmp_path)

    # Verify that the index and metadata were restored.
    assert loaded_store.size == 3
    assert len(loaded_store.chunks) == 3

    # Search again using the restored index.
    loaded_results = loaded_store.search(query, top_k=2)

    # Verify that the search results are identical.
    assert [result.chunk.chunk_id for result in loaded_results] == [
        result.chunk.chunk_id for result in original_results
    ]

    assert [result.score for result in loaded_results] == pytest.approx(
        [result.score for result in original_results]
    )

    # Verify that the original document content survived serialization.
    assert loaded_results[0].chunk.text == original_results[0].chunk.text


def test_load_missing_index_raises_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        FaissVectorStore.load(tmp_path)
