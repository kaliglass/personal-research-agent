from pathlib import Path

from app.retrieval.embeddings import BGEEmbedder
from app.retrieval.retriever import Retriever
from app.retrieval.vector_store import FaissVectorStore


def main() -> None:
    # 1. Load the BGE embedding model.
    embedder = BGEEmbedder()

    # 2. Load the previously saved FAISS index.
    storage_path = Path("data/index/transformer-paper")

    vector_store = FaissVectorStore.load(storage_path)

    print("FAISS INDEX SIZE:", vector_store.size)

    # 3. Create the Retriever.
    retriever = Retriever(
        embedder=embedder,
        vector_store=vector_store,
    )

    # 4. Ask a research question.
    question = (
        "Why did the authors introduce self-attention "
        "instead of recurrent neural networks?"
    )

    # 5. Retrieve the top three passages.
    results = retriever.retrieve(
        query=question,
        top_k=3,
    )

    print("\nQUESTION:", question)

    for rank, result in enumerate(results, start=1):
        chunk = result.chunk

        print("\n" + "=" * 60)
        print("RANK:", rank)
        print("SIMILARITY:", round(result.score, 4))
        print("DOCUMENT:", chunk.document_name)
        print("PAGE:", chunk.page_number)
        print("CHUNK ID:", chunk.chunk_id)
        print("\nPASSAGE:\n")
        print(chunk.text[:800])


if __name__ == "__main__":
    main()
