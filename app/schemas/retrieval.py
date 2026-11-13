from pydantic import BaseModel, ConfigDict

from app.schemas.chunk import DocumentChunk


class SearchResult(BaseModel):
    """A retrieved document chunk and its similarity score."""

    model_config = ConfigDict(extra="forbid")

    chunk: DocumentChunk
    score: float
