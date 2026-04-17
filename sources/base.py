"""DataSource Protocol — the contract every source plugin must satisfy."""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

from llama_index.core import Document


@runtime_checkable
class DataSource(Protocol):
    """A plugin that yields LlamaIndex Documents.

    Implementations should set `doc.doc_id` to a stable, deduplicable identifier
    (e.g. an email's Message-Id, a file path, an image SHA). The pipeline uses
    `doc_id` to skip documents that have already been ingested.
    """

    #: A short, stable name used as the default ChromaDB collection.
    collection_name: str

    def load(self, **kwargs: Any) -> list[Document]:
        """Return all documents this source exposes for the given kwargs."""
        ...
