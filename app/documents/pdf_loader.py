from pathlib import Path

from pypdf import PdfReader

from app.schemas.document import (
    PDFDocument,
    PDFMetadata,
    PDFPage,
)


class PDFLoaderError(Exception):
    """Base exception for PDF loading failures."""


class EncryptedPDFError(PDFLoaderError):
    """Raised when a PDF requires password handling."""


class PDFTextExtractionError(PDFLoaderError):
    """Raised when text cannot be extracted from a PDF."""


def _clean_metadata(value: object | None) -> str | None:
    """Convert PDF metadata values into clean optional strings."""

    if value is None:
        return None

    cleaned_value = str(value).strip()

    return cleaned_value or None


def load_pdf(file_path: str | Path) -> PDFDocument:
    """
    Load a PDF and extract its metadata and page-level text.

    Args:
        file_path: Path to the PDF file.

    Returns:
        A validated PDFDocument.

    Raises:
        FileNotFoundError: If the PDF does not exist.
        ValueError: If the path is not a PDF file.
        EncryptedPDFError: If the PDF is encrypted.
        PDFTextExtractionError: If no text can be extracted.
        PDFLoaderError: If the PDF cannot be processed.
    """

    pdf_path = Path(file_path).expanduser()

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError("The supplied file must have a .pdf extension")

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    if not pdf_path.is_file():
        raise ValueError(f"Not a file: {pdf_path}")

    try:
        with PdfReader(pdf_path) as reader:
            if reader.is_encrypted:
                raise EncryptedPDFError(f"PDF is encrypted: {pdf_path.name}")

            extracted_pages = []
            has_extractable_text = False

            for page_number, page in enumerate(reader.pages, start=1):
                try:
                    text = page.extract_text() or ""

                except Exception as exc:
                    raise PDFTextExtractionError(
                        f"Text extraction failed on page {page_number}"
                    ) from exc

                text = text.strip()

                if text:
                    has_extractable_text = True

                extracted_pages.append(
                    PDFPage(
                        page_number=page_number,
                        text=text,
                    )
                )

            if not has_extractable_text:
                raise PDFTextExtractionError(
                    f"No extractable text found in {pdf_path.name}. "
                    "The PDF may be scanned or image-based."
                )

            raw_metadata = reader.metadata

            metadata = PDFMetadata(
                title=_clean_metadata(raw_metadata.title if raw_metadata else None),
                author=_clean_metadata(raw_metadata.author if raw_metadata else None),
                subject=_clean_metadata(raw_metadata.subject if raw_metadata else None),
                creator=_clean_metadata(raw_metadata.creator if raw_metadata else None),
                producer=_clean_metadata(
                    raw_metadata.producer if raw_metadata else None
                ),
            )

            document = PDFDocument(
                filename=pdf_path.name,
                source_path=str(pdf_path),
                file_size_bytes=pdf_path.stat().st_size,
                metadata=metadata,
                pages=extracted_pages,
            )

        return document

    except PDFLoaderError:
        raise

    except Exception as exc:
        raise PDFLoaderError(f"Failed to process PDF '{pdf_path.name}': {exc}") from exc
