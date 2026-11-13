from types import SimpleNamespace

import pytest

from app.llm.grounded_generator import GroundedGenerator
from app.schemas.chunk import DocumentChunk
from app.schemas.retrieval import SearchResult


class FakeInteractions:
    """Simulate Gemini Interactions API responses."""

    def __init__(self, output_text):
        self.output_text = output_text
        self.received_request = None
        self.called = False

    def create(self, **kwargs):
        self.called = True
        self.received_request = kwargs

        return SimpleNamespace(output_text=self.output_text)


@pytest.fixture
def sample_results():
    chunks = [
        DocumentChunk(
            chunk_id=f"paper.pdf:page-{page}:chunk-0",
            document_name="paper.pdf",
            source_path="data/library/paper.pdf",
            page_number=page,
            chunk_index=0,
            token_count=10,
            text=text,
        )
        for page, text in [
            (1, "The Transformer uses attention mechanisms."),
            (2, "Self-attention allows parallel computation."),
        ]
    ]

    return [
        SearchResult(chunk=chunk, score=score)
        for chunk, score in zip(chunks, [0.9, 0.85])
    ]


def test_generate_uses_retrieved_evidence(sample_results):
    fake_api = FakeInteractions(
        "The Transformer uses attention [S1]. "
        "Self-attention supports parallel computation [S2]."
    )

    generator = GroundedGenerator(
        create_interaction=fake_api.create,
        model="test-model",
    )

    answer = generator.generate(
        "Why does the Transformer use self-attention?",
        sample_results,
    )

    assert fake_api.called
    assert "Why does the Transformer" in (fake_api.received_request["input"])
    assert (
        "The Transformer uses attention mechanisms."
        in (fake_api.received_request["input"])
    )

    assert "[S1]" in answer
    assert "[S2]" in answer

    assert "paper.pdf, page 1" in answer
    assert "paper.pdf, page 2" in answer


def test_generate_returns_message_when_no_evidence_exists():
    fake_api = FakeInteractions("Unused response")

    generator = GroundedGenerator(
        create_interaction=fake_api.create,
        model="test-model",
    )

    answer = generator.generate(
        "Explain the Transformer",
        [],
    )

    assert "could not find relevant evidence" in answer
    assert not fake_api.called


def test_generate_rejects_invalid_citations(sample_results):
    fake_api = FakeInteractions("The Transformer uses attention [S9].")

    generator = GroundedGenerator(
        create_interaction=fake_api.create,
        model="test-model",
    )

    with pytest.raises(RuntimeError):
        generator.generate(
            "Explain attention",
            sample_results,
        )


def test_generate_rejects_empty_question():
    fake_api = FakeInteractions("Unused response")

    generator = GroundedGenerator(
        create_interaction=fake_api.create,
        model="test-model",
    )

    with pytest.raises(ValueError):
        generator.generate("   ", [])

    assert not fake_api.called


def test_generate_rejects_empty_model_name():
    fake_api = FakeInteractions("Unused response")

    with pytest.raises(ValueError):
        GroundedGenerator(
            create_interaction=fake_api.create,
            model="",
        )
