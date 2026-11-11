from pathlib import Path

import pytest

from app.documents.pdf_loader import load_pdf

PDF_LIBRARY = Path("data/library")

PDF_FILES = sorted(PDF_LIBRARY.glob("*.pdf"))

if not PDF_FILES:
    pytest.skip(
        "No research PDFs found in data/library",
        allow_module_level=True,
    )


@pytest.mark.parametrize(
    "pdf_path",
    PDF_FILES,
    ids=[path.name for path in PDF_FILES],
)
def test_real_research_pdf_extraction(pdf_path):
    """Verify extraction against actual research PDFs."""

    document = load_pdf(pdf_path)

    assert document.filename == pdf_path.name
    assert document.page_count > 0
    assert len(document.pages) == document.page_count

    # Page numbers must be sequential, starting at 1.
    for expected_number, page in enumerate(document.pages, start=1):
        assert page.page_number == expected_number

    # The document must contain some extractable text.
    total_characters = sum(len(page.text) for page in document.pages)

    assert total_characters > 0

    print(f"\nDocument: {document.filename}")
    print(f"Pages: {document.page_count}")
    print(f"Extracted characters: {total_characters}")
