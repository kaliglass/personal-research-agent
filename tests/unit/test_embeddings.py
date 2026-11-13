import numpy as np
import pytest

from app.retrieval.embeddings import BGEEmbedder
from app.schemas.chunk import DocumentChunk


@pytest.fixture
def fake_embedder(monkeypatch):
    class FakeSentenceTransformer:
        def __init__(self, model_name, device=None):
            self.device = device or "cpu"
            self.received_texts = []
            self.encode_called = False

        def get_sentence_embedding_dimension(self):
            return 768

        def encode(self, texts, **kwargs):
            self.encode_called = True
            self.received_texts = texts

            vectors = np.zeros((len(texts), 768), dtype=np.float32)
            vectors[:, 0] = 1.0

            return vectors

    monkeypatch.setattr(
        "app.retrieval.embeddings.SentenceTransformer",
        FakeSentenceTransformer,
    )

    embedder = BGEEmbedder()
    return embedder, embedder.model


def test_embedding_output_shape_and_normalization(fake_embedder):
    embedder, _ = fake_embedder

    vectors = embedder.embed_texts(
        [
            "The Transformer uses attention.",
            "Neural networks learn representations.",
        ]
    )

    assert vectors.shape == (2, 768)
    assert vectors.dtype == np.float32

    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms, 1.0)


def test_empty_input_returns_empty_matrix(fake_embedder):
    embedder, model = fake_embedder

    vectors = embedder.embed_texts([])

    assert vectors.shape == (0, 768)
    assert not model.encode_called


def test_single_string_is_rejected(fake_embedder):
    embedder, _ = fake_embedder

    with pytest.raises(TypeError):
        embedder.embed_texts("This is not a sequence of texts")


def test_empty_text_is_rejected(fake_embedder):
    embedder, _ = fake_embedder

    with pytest.raises(ValueError):
        embedder.embed_texts(["valid text", "   "])


def test_invalid_batch_size_is_rejected(fake_embedder):
    embedder, _ = fake_embedder

    with pytest.raises(ValueError):
        embedder.embed_texts(["some text"], batch_size=0)


def test_embed_chunks_preserves_text_order(fake_embedder):
    embedder, model = fake_embedder

    chunks = [
        DocumentChunk(
            chunk_id=f"doc:page-1:chunk-{i}",
            document_name="test.pdf",
            source_path="data/library/test.pdf",
            page_number=1,
            chunk_index=i,
            token_count=3,
            text=text,
        )
        for i, text in enumerate(
            [
                "First document passage",
                "Second document passage",
            ]
        )
    ]

    vectors = embedder.embed_chunks(chunks)

    assert vectors.shape == (2, 768)
    assert model.received_texts == [
        "First document passage",
        "Second document passage",
    ]
