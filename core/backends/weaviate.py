"""Weaviate vector backend — stub.

To activate:

1. `pip install weaviate-client` (already in requirements.txt) and
   `pip install llama-index-vector-stores-weaviate`.
2. Set these in `.env`:
     WEAVIATE_URL=...          # e.g. https://my-cluster.weaviate.network
     WEAVIATE_API_KEY=...      # only for Weaviate Cloud; omit for local
     VECTOR_BACKEND=weaviate
3. Replace the `NotImplementedError` raises below with real calls using
   `import weaviate` and
   `from llama_index.vector_stores.weaviate import WeaviateVectorStore`.
   Return the `WeaviateVectorStore` from `get_vector_store`.
"""
from __future__ import annotations

import config
from llama_index.core.vector_stores.types import BasePydanticVectorStore


_SETUP_MSG = (
    "WeaviateBackend is a stub. Set WEAVIATE_URL (and WEAVIATE_API_KEY for "
    "Weaviate Cloud) in .env, install `llama-index-vector-stores-weaviate`, "
    "and fill in core/backends/weaviate.py."
)


class WeaviateBackend:
    """Stubbed Weaviate backend — raises with setup instructions on use."""

    def __init__(self) -> None:
        if not config.WEAVIATE_URL:
            raise NotImplementedError(
                f"{_SETUP_MSG} Missing env var: WEAVIATE_URL."
            )
        raise NotImplementedError(_SETUP_MSG)

    def get_vector_store(self, collection: str) -> BasePydanticVectorStore:
        raise NotImplementedError(_SETUP_MSG)

    def collection_exists(self, collection: str) -> bool:
        raise NotImplementedError(_SETUP_MSG)

    def existing_doc_ids(self, collection: str) -> set[str]:
        raise NotImplementedError(_SETUP_MSG)

    def delete_collection(self, collection: str) -> None:
        raise NotImplementedError(_SETUP_MSG)
