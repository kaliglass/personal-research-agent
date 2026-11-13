from pathlib import Path

import numpy as np

from app.documents.chunker import (
    chunk_document,
    load_bge_tokenizer,
)
from app.documents.pdf_loader import load_pdf
from app.retrieval.embeddings import BGEEmbedder


def main() -> None:
    pdf_files = sorted(Path("data/library").glob("*.pdf"))

    if len(pdf_files) != 1:
        raise SystemExit(f"Expected exactly one PDF, found {len(pdf_files)}")

    # Step 1: Extract the PDF.
    document = load_pdf(pdf_files[0])

    # Step 2: Divide it into chunks.
    tokenizer = load_bge_tokenizer()

    chunks = chunk_document(
        document=document,
        tokenizer=tokenizer,
        chunk_size=300,
        overlap=50,
    )

    # Step 3: Load BGE and generate embeddings.
    embedder = BGEEmbedder()

    embeddings = embedder.embed_chunks(chunks)

    # Step 4: Inspect the result.
    norms = np.linalg.norm(embeddings, axis=1)

    print("\nDOCUMENT:", document.filename)
    print("TOTAL CHUNKS:", len(chunks))
    print("MODEL:", "BAAI/bge-base-en-v1.5")
    print("DEVICE:", embedder.device)
    print("EMBEDDING SHAPE:", embeddings.shape)
    print("DATA TYPE:", embeddings.dtype)

    print("\nFIRST CHUNK:")
    print(chunks[0].text[:300])

    print("\nFIRST VECTOR (10 VALUES):")
    print(embeddings[0][:10])

    print("\nFIRST VECTOR NORM:")
    print(norms[0])

    print("\nALL VECTORS NORMALIZED:")
    print(np.allclose(norms, 1.0, atol=1e-3))


if __name__ == "__main__":
    main()
