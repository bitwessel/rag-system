"""RAG pipeline: wires a DataSource + vector backend + OpenRouter LLM/embedder."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter

import config
from core.backends import VectorBackend, build_backend
from core.embeddings import OpenRouterEmbedding
from core.llm import OpenRouterLLM
from sources.base import DataSource


@dataclass
class RAGResponse:
    """Structured response returned by `RAGPipeline.query`."""

    answer: str
    sources: list[dict[str, Any]] = field(default_factory=list)
    tokens_used: int | None = None


class RAGPipeline:
    """End-to-end RAG pipeline for a single DataSource + collection."""

    def __init__(
        self,
        source: DataSource,
        collection_name: str | None = None,
        backend: VectorBackend | None = None,
        llm: OpenRouterLLM | None = None,
        embed_model: OpenRouterEmbedding | None = None,
    ) -> None:
        self.source = source
        self.collection_name = collection_name or getattr(
            source, "collection_name", "documents"
        )
        self.backend: VectorBackend = backend or build_backend(config.VECTOR_BACKEND)
        self.embed_model: OpenRouterEmbedding = embed_model or OpenRouterEmbedding()
        self.llm: OpenRouterLLM = llm or OpenRouterLLM()

        self._vector_store = self.backend.get_vector_store(self.collection_name)
        self._storage_context = StorageContext.from_defaults(
            vector_store=self._vector_store
        )
        self._splitter = SentenceSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
        )
        self._index: VectorStoreIndex | None = None

    # ----- public API ---------------------------------------------------------

    def ingest(self, **source_kwargs: Any) -> int:
        """Load documents from the source and add newly-seen ones to the index.

        Dedupes against any `doc_id`s already present in the collection.
        Returns the number of documents newly inserted.
        """
        docs: list[Document] = self.source.load(**source_kwargs)
        if not docs:
            return 0

        existing = self.backend.existing_doc_ids(self.collection_name)
        fresh = [d for d in docs if d.doc_id not in existing]
        if not fresh:
            self._index = self._get_or_build_index()
            return 0

        index = self._get_or_build_index()
        nodes = self._splitter.get_nodes_from_documents(fresh)
        for node in nodes:
            parent = node.ref_doc_id
            if parent:
                node.metadata = {**(node.metadata or {}), "doc_id": parent}
        index.insert_nodes(nodes)
        return len(fresh)

    def query(self, question: str, top_k: int = 5) -> RAGResponse:
        """Run retrieval + generation for a question."""
        index = self._get_or_build_index()
        engine = index.as_query_engine(
            llm=self.llm,
            similarity_top_k=top_k,
        )
        result = engine.query(question)
        sources = [
            {
                **(node.metadata or {}),
                "score": getattr(node, "score", None),
            }
            for node in getattr(result, "source_nodes", []) or []
        ]
        return RAGResponse(
            answer=str(result),
            sources=sources,
            tokens_used=None,
        )

    # ----- internals ----------------------------------------------------------

    def _get_or_build_index(self) -> VectorStoreIndex:
        if self._index is not None:
            return self._index
        self._index = VectorStoreIndex.from_vector_store(
            vector_store=self._vector_store,
            embed_model=self.embed_model,
            storage_context=self._storage_context,
        )
        return self._index
