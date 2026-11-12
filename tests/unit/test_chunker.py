from app.documents.chunker import chunk_document
from app.schemas.document import PDFDocument, PDFMetadata, PDFPage


class FakeTokenizer:
    """A lightweight tokenizer substitute for unit tests."""

    def __call__(
        self,
        text,
        add_special_tokens=False,
        return_offsets_mapping=False,
    ):
        offsets = [
            (match.start(), match.end())
            for match in __import__("re").finditer(r"\S+", text)
        ]

        return {"offset_mapping": offsets}


def make_document(pages: list[PDFPage]) -> PDFDocument:
    return PDFDocument(
        filename="test.pdf",
        source_path="data/library/test.pdf",
        file_size_bytes=100,
        metadata=PDFMetadata(),
        pages=pages,
    )


def test_chunks_respect_chunk_size_and_overlap():
    document = make_document(
        [
            PDFPage(
                page_number=1,
                text="one two three four five six seven eight nine ten",
            )
        ]
    )

    chunks = chunk_document(
        document,
        FakeTokenizer(),
        chunk_size=4,
        overlap=1,
    )

    assert len(chunks) == 3
    assert chunks[0].text == "one two three four"
    assert chunks[1].text == "four five six seven"
    assert chunks[2].text == "seven eight nine ten"


def test_chunks_preserve_original_text():
    original_text = "Attention Is All You Need. This is the Transformer paper."

    document = make_document([PDFPage(page_number=1, text=original_text)])

    chunks = chunk_document(
        document,
        FakeTokenizer(),
        chunk_size=100,
        overlap=10,
    )

    assert chunks[0].text == original_text


def test_chunk_metadata_is_preserved():
    document = make_document([PDFPage(page_number=3, text="one two three four five")])

    chunks = chunk_document(
        document,
        FakeTokenizer(),
        chunk_size=3,
        overlap=1,
    )

    assert chunks[0].document_name == "test.pdf"
    assert chunks[0].page_number == 3
    assert chunks[0].chunk_index == 0
    assert chunks[0].token_count == 3


def test_empty_pages_are_skipped():
    document = make_document(
        [
            PDFPage(page_number=1, text="   "),
            PDFPage(page_number=2, text="one two three"),
        ]
    )

    chunks = chunk_document(
        document,
        FakeTokenizer(),
        chunk_size=10,
        overlap=2,
    )

    assert len(chunks) == 1
    assert chunks[0].page_number == 2


def test_invalid_chunk_size_is_rejected():
    document = make_document([PDFPage(page_number=1, text="one two three")])

    import pytest

    with pytest.raises(ValueError):
        chunk_document(document, FakeTokenizer(), chunk_size=0)


def test_invalid_overlap_is_rejected():
    document = make_document([PDFPage(page_number=1, text="one two three")])

    import pytest

    with pytest.raises(ValueError):
        chunk_document(document, FakeTokenizer(), chunk_size=3, overlap=3)
