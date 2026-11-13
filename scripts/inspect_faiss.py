from pathlib import Path

from app.documents.chunker import (
    chunk_document,
    load_bge_tokenizer,
)
from app.documents.pdf_loader import load_pdf
from app.retrieval.embeddings import BGEEmbedder
from app.retrieval.vector_store import FaissVectorStore


def main() -> None:
    pdf_files = sorted(Path("data/library").glob("*.pdf"))

    if len(pdf_files) != 1:
        raise SystemExit(f"Expected exactly one PDF, found {len(pdf_files)}")

    # 1. Load the PDF.
    document = load_pdf(pdf_files[0])

    # 2. Generate document chunks.
    tokenizer = load_bge_tokenizer()

    chunks = chunk_document(
        document=document,
        tokenizer=tokenizer,
        chunk_size=300,
        overlap=50,
    )

    # 3. Generate document embeddings.
    embedder = BGEEmbedder()
    document_embeddings = embedder.embed_chunks(chunks)

    # 4. Build the FAISS index.
    vector_store = FaissVectorStore(embedding_dimension=embedder.embedding_dimension)

    vector_store.add(chunks, document_embeddings)

    print("\nDOCUMENT:", document.filename)
    print("CHUNKS:", len(chunks))
    print("FAISS INDEX SIZE:", vector_store.size)

    # 5. Save the FAISS index and chunk metadata.
    storage_path = Path("data/index/transformer-paper")

    vector_store.save(storage_path)

    print("\nINDEX SAVED:", storage_path)

    # 6. Load a new vector store from disk.
    loaded_store = FaissVectorStore.load(storage_path)

    print("RELOADED INDEX SIZE:", loaded_store.size)

    # 7. Convert the question into a query embedding.
    question = (
        "Why did the authors introduce self-attention "
        "instead of recurrent neural networks?"
    )

    query_embedding = embedder.embed_query(question)

    # 8. Search using the restored FAISS index.
    results = loaded_store.search(
        query_embedding,
        top_k=3,
    )

    print("\nQUESTION:", question)

    for rank, result in enumerate(results, start=1):
        chunk = result.chunk

        print("\n" + "=" * 60)
        print("RANK:", rank)
        print("SIMILARITY:", round(result.score, 4))
        print("CHUNK ID:", chunk.chunk_id)
        print("PAGE:", chunk.page_number)
        print("\nPASSAGE:\n")
        print(chunk.text[:800])


if __name__ == "__main__":
    main()
