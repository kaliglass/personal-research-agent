from pydantic import BaseModel, ConfigDict, Field


class DocumentChunk(BaseModel):
    """A text passage extracted from a source document."""

    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    document_name: str
    source_path: str
    page_number: int = Field(ge=1)
    chunk_index: int = Field(ge=0)
    token_count: int = Field(ge=1)
    text: str = Field(min_length=1)
