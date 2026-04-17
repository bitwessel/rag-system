"""Pinecone vector backend — stub.

To activate:

1. `pip install pinecone-client` (already in requirements.txt).
2. Set these in `.env`:
     PINECONE_API_KEY=...
     PINECONE_ENVIRONMENT=...   # e.g. "us-east-1-aws" or the Pinecone region
     PINECONE_INDEX=...         # the index name to use
     VECTOR_BACKEND=pinecone
3. Replace the `NotImplementedError` raises below with real calls using
   `from pinecone import Pinecone` and
   `from llama_index.vector_stores.pinecone import PineconeVectorStore`
   (install `llama-index-vector-stores-pinecone`). Return the
   `PineconeVectorStore` from `get_vector_store`.
"""
from __future__ import annotations

import config
from llama_index.core.vector_stores.types import BasePydanticVectorStore


_SETUP_MSG = (
    "PineconeBackend is a stub. Set PINECONE_API_KEY, PINECONE_ENVIRONMENT, "
    "PINECONE_INDEX in .env, install `llama-index-vector-stores-pinecone`, "
    "and fill in core/backends/pinecone.py."
)


class PineconeBackend:
    """Stubbed Pinecone backend — raises with setup instructions on use."""

    def __init__(self) -> None:
        missing = [
            k
            for k, v in {
                "PINECONE_API_KEY": config.PINECONE_API_KEY,
                "PINECONE_ENVIRONMENT": config.PINECONE_ENVIRONMENT,
                "PINECONE_INDEX": config.PINECONE_INDEX,
            }.items()
            if not v
        ]
        if missing:
            raise NotImplementedError(
                f"{_SETUP_MSG} Missing env vars: {', '.join(missing)}."
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
