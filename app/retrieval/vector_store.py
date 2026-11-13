import json
from collections.abc import Sequence
from pathlib import Path
from uuid import uuid4

import faiss
import numpy as np

from app.schemas.chunk import DocumentChunk
from app.schemas.retrieval import SearchResult

PERSISTENCE_FORMAT_VERSION = 1


class FaissVectorStore:
    """An exact cosine-similarity vector store backed by FAISS."""

    def __init__(self, embedding_dimension: int) -> None:
        if embedding_dimension <= 0:
            raise ValueError("Embedding dimension must be positive")

        self.embedding_dimension = embedding_dimension

        # Create an exact inner-product index.
        self.index = faiss.IndexFlatIP(embedding_dimension)

        # Keep document chunks aligned with FAISS insertion positions.
        self.chunks: list[DocumentChunk] = []
        self.chunk_ids: set[str] = set()

    @property
    def size(self) -> int:
        return self.index.ntotal

    def add(
        self,
        chunks: Sequence[DocumentChunk],
        embeddings: np.ndarray,
    ) -> None:
        """Add document vectors and their corresponding metadata."""

        if not chunks:
            if np.asarray(embeddings).size != 0:
                raise ValueError("Received embeddings without chunks")
            return

        vectors = np.asarray(embeddings, dtype=np.float32)

        if vectors.ndim != 2:
            raise ValueError("Embeddings must be a 2D matrix")

        if vectors.shape[0] != len(chunks):
            raise ValueError("Number of embeddings must match number of chunks")

        if vectors.shape[1] != self.embedding_dimension:
            raise ValueError(
                f"Expected {self.embedding_dimension} dimensions, "
                f"got {vectors.shape[1]}"
            )

        if not np.isfinite(vectors).all():
            raise ValueError("Embeddings contain invalid values")

        norms = np.linalg.norm(vectors, axis=1)

        if not np.allclose(norms, 1.0, atol=1e-3):
            raise ValueError("All document embeddings must be normalized")

        incoming_ids = [chunk.chunk_id for chunk in chunks]

        if len(incoming_ids) != len(set(incoming_ids)):
            raise ValueError("Duplicate chunk IDs in input")

        if self.chunk_ids.intersection(incoming_ids):
            raise ValueError("Some chunks have already been indexed")

        # FAISS expects contiguous float32 matrices.
        vectors = np.ascontiguousarray(vectors)

        # Insert vectors first, then maintain their metadata mapping.
        self.index.add(vectors)
        self.chunks.extend(chunks)
        self.chunk_ids.update(incoming_ids)

    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5,
    ) -> list[SearchResult]:
        """Find the most similar document chunks for one query vector."""

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        query = np.asarray(query_embedding, dtype=np.float32)

        if query.ndim == 1:
            query = query.reshape(1, -1)

        if query.ndim != 2 or query.shape[0] != 1:
            raise ValueError("Expected exactly one query vector")

        if query.shape[1] != self.embedding_dimension:
            raise ValueError(
                f"Expected {self.embedding_dimension} dimensions, got {query.shape[1]}"
            )

        if not np.isfinite(query).all():
            raise ValueError("Query contains invalid values")

        query_norm = np.linalg.norm(query, axis=1)

        if not np.allclose(query_norm, 1.0, atol=1e-3):
            raise ValueError("Query embedding must be normalized")

        if self.size == 0:
            return []

        query = np.ascontiguousarray(query)

        # Do not request more results than the index contains.
        k = min(top_k, self.size)

        scores, indices = self.index.search(query, k)

        results = []

        for score, index_id in zip(scores[0], indices[0]):
            if index_id < 0:
                continue

            results.append(
                SearchResult(
                    chunk=self.chunks[int(index_id)],
                    score=float(score),
                )
            )

        return results

    def save(self, directory: str | Path) -> None:
        """
        Persist the FAISS index and its corresponding chunk metadata.
        """

        if self.size != len(self.chunks):
            raise RuntimeError("FAISS index size does not match chunk metadata count")

        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        # Every save gets a unique snapshot ID.
        snapshot_id = uuid4().hex

        index_filename = f"faiss-{snapshot_id}.index"
        metadata_filename = f"chunks-{snapshot_id}.json"

        index_path = directory / index_filename
        metadata_path = directory / metadata_filename

        # Temporary files are written before the manifest is updated.
        index_temp = directory / f".{index_filename}.tmp"
        metadata_temp = directory / f".{metadata_filename}.tmp"
        manifest_temp = directory / ".manifest.json.tmp"

        try:
            # 1. Save the numerical FAISS index.
            faiss.write_index(self.index, str(index_temp))

            # 2. Save the corresponding original document chunks.
            metadata = {
                "format_version": PERSISTENCE_FORMAT_VERSION,
                "snapshot_id": snapshot_id,
                "embedding_dimension": self.embedding_dimension,
                "chunks": [chunk.model_dump(mode="json") for chunk in self.chunks],
            }

            metadata_temp.write_text(
                json.dumps(metadata, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

            # 3. Move both completed files into their final locations.
            index_temp.replace(index_path)
            metadata_temp.replace(metadata_path)

            # 4. Publish the snapshot only after both files exist.
            manifest = {
                "format_version": PERSISTENCE_FORMAT_VERSION,
                "snapshot_id": snapshot_id,
                "embedding_dimension": self.embedding_dimension,
                "index_file": index_filename,
                "metadata_file": metadata_filename,
            }

            manifest_temp.write_text(
                json.dumps(manifest, indent=2),
                encoding="utf-8",
            )

            # Atomic replacement on the same filesystem.
            manifest_temp.replace(directory / "manifest.json")

        finally:
            # Remove leftover temporary files if saving fails.
            for temporary_file in (
                index_temp,
                metadata_temp,
                manifest_temp,
            ):
                temporary_file.unlink(missing_ok=True)

    @classmethod
    def load(cls, directory: str | Path) -> "FaissVectorStore":
        """Load a previously persisted FAISS index and chunk metadata."""

        directory = Path(directory)
        manifest_path = directory / "manifest.json"

        if not manifest_path.is_file():
            raise FileNotFoundError(f"FAISS manifest not found: {manifest_path}")

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        if manifest.get("format_version") != PERSISTENCE_FORMAT_VERSION:
            raise ValueError("Unsupported FAISS persistence format version")

        snapshot_id = manifest["snapshot_id"]
        index_filename = manifest["index_file"]
        metadata_filename = manifest["metadata_file"]

        # Only allow filenames, not arbitrary paths from the manifest.
        for filename in (index_filename, metadata_filename):
            if not isinstance(filename, str) or Path(filename).name != filename:
                raise ValueError("Invalid snapshot filename in manifest")

        index_path = directory / index_filename
        metadata_path = directory / metadata_filename

        if not index_path.is_file() or not metadata_path.is_file():
            raise FileNotFoundError(
                "FAISS index or corresponding metadata file is missing"
            )

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        # Ensure both files belong to the same saved version.
        if metadata.get("snapshot_id") != snapshot_id:
            raise ValueError("Index and metadata snapshot IDs do not match")

        if metadata.get("format_version") != PERSISTENCE_FORMAT_VERSION:
            raise ValueError("Unsupported metadata format version")

        embedding_dimension = metadata["embedding_dimension"]

        if embedding_dimension != manifest["embedding_dimension"]:
            raise ValueError("Embedding dimensions do not match")

        # Reconstruct the Pydantic document chunk objects.
        chunks = [DocumentChunk.model_validate(item) for item in metadata["chunks"]]

        # Restore the FAISS index.
        index = faiss.read_index(str(index_path))

        # Validate index and metadata compatibility.
        if index.d != embedding_dimension:
            raise ValueError("FAISS index dimension does not match metadata")

        if index.metric_type != faiss.METRIC_INNER_PRODUCT:
            raise ValueError("Expected an inner-product FAISS index")

        if index.ntotal != len(chunks):
            raise ValueError("FAISS vector count does not match metadata chunk count")

        chunk_ids = {chunk.chunk_id for chunk in chunks}

        if len(chunk_ids) != len(chunks):
            raise ValueError("Duplicate chunk IDs in persisted metadata")

        # Reconstruct the application-level vector store.
        store = cls(embedding_dimension=embedding_dimension)
        store.index = index
        store.chunks = chunks
        store.chunk_ids = chunk_ids

        return store
