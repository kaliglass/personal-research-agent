from pydantic import BaseModel, ConfigDict, Field


class PDFMetadata(BaseModel):
    """Metadata extracted from a PDF."""

    model_config = ConfigDict(extra="forbid")

    title: str | None = None
    author: str | None = None
    subject: str | None = None
    creator: str | None = None
    producer: str | None = None


class PDFPage(BaseModel):
    """Text extracted from one PDF page."""

    model_config = ConfigDict(extra="forbid")

    page_number: int = Field(ge=1)
    text: str


class PDFDocument(BaseModel):
    """Structured representation of an extracted PDF."""

    model_config = ConfigDict(extra="forbid")

    filename: str
    source_path: str
    file_size_bytes: int = Field(ge=0)
    metadata: PDFMetadata
    pages: list[PDFPage] = Field(min_length=1)

    @property
    def page_count(self) -> int:
        return len(self.pages)
