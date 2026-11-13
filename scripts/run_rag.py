from pathlib import Path

from app.llm.client import MODEL_NAME, create_gemini_interaction
from app.llm.grounded_generator import GroundedGenerator
from app.retrieval.embeddings import BGEEmbedder
from app.retrieval.retriever import Retriever
from app.retrieval.vector_store import FaissVectorStore


def main() -> None:
    # 1. Load the existing BGE embedding model.
    embedder = BGEEmbedder()

    # 2. Load the persisted FAISS index.
    storage_path = Path("data/index/transformer-paper")

    vector_store = FaissVectorStore.load(storage_path)

    print("FAISS INDEX SIZE:", vector_store.size)

    # 3. Create the Retriever.
    retriever = Retriever(
        embedder=embedder,
        vector_store=vector_store,
    )

    # 4. Create the grounded generator.
    generator = GroundedGenerator(
        create_interaction=create_gemini_interaction,
        model=MODEL_NAME,
    )

    # 5. Accept a research question.
    question = (
        "Why did the authors introduce self-attention "
        "instead of recurrent neural networks?"
    )

    # 6. Retrieve relevant evidence.
    results = retriever.retrieve(
        question,
        top_k=3,
    )

    print("\nQUESTION:", question)
    print("\nRETRIEVED PASSAGES:")

    for result in results:
        print(f"\nPage {result.chunk.page_number} | Similarity: {result.score:.4f}")
        print(result.chunk.text[:300])

    # 7. Generate a grounded answer.
    answer = generator.generate(
        question=question,
        results=results,
    )

    print("\n" + "=" * 60)
    print("FINAL RESEARCH ANSWER")
    print("=" * 60)
    print(answer)


if __name__ == "__main__":
    main()
