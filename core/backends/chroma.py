"""ChromaDB vector backend (persistent, local)."""
from __future__ import annotations

from typing import TYPE_CHECKING

import chromadb
from chromadb.config import Settings
from llama_index.vector_stores.chroma import ChromaVectorStore

import config

if TYPE_CHECKING:
    from chromadb.api import ClientAPI
    from chromadb.api.models.Collection import Collection


class ChromaBackend:
    """Persistent ChromaDB backend. One collection per source type."""

    def __init__(self, persist_dir: str | None = None) -> None:
        config.ensure_data_dirs()
        self._persist_dir = persist_dir or config.CHROMA_PERSIST_DIR
        self._client: ClientAPI = chromadb.PersistentClient(
            path=self._persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )

    def _collection(self, name: str) -> Collection:
        return self._client.get_or_create_collection(name=name)

    def get_vector_store(self, collection: str) -> ChromaVectorStore:
        return ChromaVectorStore(chroma_collection=self._collection(collection))

    def collection_exists(self, collection: str) -> bool:
        return collection in [c.name for c in self._client.list_collections()]

    def existing_doc_ids(self, collection: str) -> set[str]:
        if not self.collection_exists(collection):
            return set()
        col = self._collection(collection)
        got = col.get(include=["metadatas"])
        ids: set[str] = set()
        for meta in got.get("metadatas") or []:
            if meta and "doc_id" in meta:
                ids.add(str(meta["doc_id"]))
        return ids

    def delete_collection(self, collection: str) -> None:
        if self.collection_exists(collection):
            self._client.delete_collection(name=collection)
