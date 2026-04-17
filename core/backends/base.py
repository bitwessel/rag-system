"""Vector backend Protocol.

A backend hides the details of a specific vector DB behind a small surface so
`RAGPipeline` can swap ChromaDB for Pinecone or Weaviate via one config value.
All backends return a LlamaIndex `BasePydanticVectorStore` via
`get_vector_store()` so the indexing/querying path stays the same.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from llama_index.core.vector_stores.types import BasePydanticVectorStore


@runtime_checkable
class VectorBackend(Protocol):
    """Minimal surface every vector backend must provide."""

    def get_vector_store(self, collection: str) -> BasePydanticVectorStore:
        """Return a LlamaIndex-compatible vector store for a given collection."""
        ...

    def collection_exists(self, collection: str) -> bool:
        """Return True if the named collection is already present in the backend."""
        ...

    def existing_doc_ids(self, collection: str) -> set[str]:
        """Return the set of `doc_id`s already stored in this collection.

        Used by the pipeline to skip re-ingesting previously seen documents.
        """
        ...

    def delete_collection(self, collection: str) -> None:
        """Drop a collection and all its vectors. No-op if it does not exist."""
        ...
