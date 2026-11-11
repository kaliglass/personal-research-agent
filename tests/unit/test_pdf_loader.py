from pathlib import Path

import pytest
from pypdf import PdfWriter

from app.documents.pdf_loader import (
    EncryptedPDFError,
    PDFTextExtractionError,
    load_pdf,
)


def create_blank_pdf(file_path: Path) -> None:
    """Create a valid PDF with one blank page for error testing."""

    writer = PdfWriter()
    writer.add_blank_page(width=595, height=842)

    with file_path.open("wb") as file:
        writer.write(file)


def test_missing_pdf_raises_error(tmp_path):
    """A nonexistent PDF should raise FileNotFoundError."""

    with pytest.raises(FileNotFoundError):
        load_pdf(tmp_path / "missing.pdf")


def test_non_pdf_file_is_rejected(tmp_path):
    """The loader should reject files with a non-PDF extension."""

    file_path = tmp_path / "document.txt"
    file_path.write_text("This is not a PDF")

    with pytest.raises(ValueError, match="\\.pdf"):
        load_pdf(file_path)


def test_blank_pdf_raises_extraction_error(tmp_path):
    """A PDF without extractable text should be rejected."""

    file_path = tmp_path / "blank.pdf"
    create_blank_pdf(file_path)

    with pytest.raises(PDFTextExtractionError, match="No extractable text"):
        load_pdf(file_path)


def test_encrypted_pdf_raises_error(tmp_path):
    """An encrypted PDF should be rejected."""

    file_path = tmp_path / "encrypted.pdf"

    writer = PdfWriter()
    writer.add_blank_page(width=595, height=842)
    writer.encrypt("test-password")

    with file_path.open("wb") as file:
        writer.write(file)

    with pytest.raises(EncryptedPDFError):
        load_pdf(file_path)
