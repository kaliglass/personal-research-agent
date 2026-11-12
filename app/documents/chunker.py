from transformers import AutoTokenizer, PreTrainedTokenizerBase

from app.schemas.chunk import DocumentChunk
from app.schemas.document import PDFDocument

BGE_MODEL_NAME = "BAAI/bge-base-en-v1.5"


def load_bge_tokenizer() -> PreTrainedTokenizerBase:
    """Load the tokenizer associated with our BGE embedding model."""

    return AutoTokenizer.from_pretrained(BGE_MODEL_NAME)


def chunk_document(
    document: PDFDocument,
    tokenizer: PreTrainedTokenizerBase,
    chunk_size: int = 300,
    overlap: int = 50,
) -> list[DocumentChunk]:
    """
    Divide PDF text into overlapping token-based chunks.

    Character offsets preserve the original extracted text.
    Each page is chunked independently.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be non-negative and smaller than chunk_size")

    chunks = []
    step = chunk_size - overlap

    for page in document.pages:
        text = page.text

        if not text.strip():
            continue

        encoded = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )

        offsets = encoded["offset_mapping"]
        page_chunk_index = 0

        for start in range(0, len(offsets), step):
            end = min(start + chunk_size, len(offsets))

            if start >= end:
                break

            char_start = offsets[start][0]
            char_end = offsets[end - 1][1]

            chunk_text = text[char_start:char_end].strip()

            if chunk_text:
                chunks.append(
                    DocumentChunk(
                        chunk_id=(
                            f"{document.filename}:"
                            f"page-{page.page_number}:"
                            f"chunk-{page_chunk_index}"
                        ),
                        document_name=document.filename,
                        source_path=document.source_path,
                        page_number=page.page_number,
                        chunk_index=page_chunk_index,
                        token_count=end - start,
                        text=chunk_text,
                    )
                )
                page_chunk_index += 1

            if end >= len(offsets):
                break

    return chunks
