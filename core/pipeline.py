"""RAG pipeline: wires a DataSource + vector backend + LLM/embedder."""
from __future__ import annotations

import pickle
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.base.embeddings.base import BaseEmbedding
from llama_index.core.llms import LLM
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import MetadataMode
from tqdm import tqdm

import config
from core.backends import VectorBackend, build_backend
from sources.base import DataSource


def _build_llm(model_override: str | None = None) -> LLM:
    if config.LLM_SOURCE == "ollama":
        from llama_index.llms.ollama import Ollama
        model = model_override or config.OLLAMA_LLM_MODEL
        return Ollama(model=model, base_url=config.OLLAMA_BASE_URL, request_timeout=300.0, context_window=4096)
    from core.llm import OpenRouterLLM
    return OpenRouterLLM()


def _build_embed_model() -> BaseEmbedding:
    if config.EMBED_SOURCE == "ollama":
        from llama_index.embeddings.ollama import OllamaEmbedding
        return OllamaEmbedding(model_name=config.OLLAMA_EMBED_MODEL, base_url=config.OLLAMA_BASE_URL)
    from core.embeddings import OpenRouterEmbedding
    return OpenRouterEmbedding()


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
        llm: LLM | None = None,
        embed_model: BaseEmbedding | None = None,
        llm_model_override: str | None = None,
    ) -> None:
        self.source = source
        self.collection_name = collection_name or getattr(
            source, "collection_name", "documents"
        )
        self.backend: VectorBackend = backend or build_backend(config.VECTOR_BACKEND)
        self.embed_model: BaseEmbedding = embed_model or _build_embed_model()
        self.llm: LLM = llm or _build_llm(llm_model_override)

        self._vector_store = self.backend.get_vector_store(self.collection_name)
        self._storage_context = StorageContext.from_defaults(
            vector_store=self._vector_store
        )
        self._splitter = SentenceSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            include_metadata=False,
        )
        self._index: VectorStoreIndex | None = None

    # ----- public API ---------------------------------------------------------

    def ingest(self, fresh: bool = False, **source_kwargs: Any) -> int:
        """Load documents from the source and add newly-seen ones to the index.

        Dedupes against any `doc_id`s already present in the collection.
        Pass ``fresh=True`` to wipe the collection before ingesting.
        Returns the number of documents newly inserted.
        """
        t_start = time.monotonic()

        if fresh:
            tqdm.write(f"[pipeline] Dropping collection '{self.collection_name}'...")
            self.backend.delete_collection(self.collection_name)
            self._index = None
            self._vector_store = self.backend.get_vector_store(self.collection_name)
            self._storage_context = StorageContext.from_defaults(
                vector_store=self._vector_store
            )

        cache_path = Path(config.NODE_CACHE_PATH)
        new_doc_count = 0

        if cache_path.exists():
            tqdm.write(f"[pipeline] Loading node cache from {cache_path} (skipping ingestion)...")
            with open(cache_path, "rb") as f:
                nodes = pickle.load(f)
            tqdm.write(f"[pipeline] Loaded {len(nodes)} cached node(s)")
        else:
            tqdm.write("[pipeline] Checking for already-indexed documents...")
            existing = self.backend.existing_doc_ids(self.collection_name)
            if existing:
                tqdm.write(
                    f"[pipeline] {len(existing)} document(s) already indexed; "
                    f"passing to source for pre-filtering..."
                )
                source_kwargs["skip_ids"] = existing

            tqdm.write("[pipeline] Loading documents from source...")
            docs: list[Document] = self.source.load(**source_kwargs)
            tqdm.write(f"[pipeline] Source returned {len(docs)} document(s)")
            if not docs:
                return 0

            new_docs = [d for d in docs if d.doc_id not in existing]
            skipped = len(docs) - len(new_docs)
            if skipped:
                tqdm.write(f"[pipeline] Skipping {skipped} already-present document(s)")
            if not new_docs:
                self._index = self._get_or_build_index()
                return 0

            tqdm.write(f"[pipeline] Splitting {len(new_docs)} document(s) into chunks...")
            doc_meta_by_id = {d.doc_id: (d.metadata or {}) for d in new_docs}
            nodes = self._splitter.get_nodes_from_documents(new_docs)
            for node in nodes:
                parent = node.ref_doc_id
                parent_meta = doc_meta_by_id.get(parent, {}) if parent else {}
                existing = node.metadata or {}
                node.metadata = {
                    **parent_meta,
                    **existing,
                    "doc_id": parent or existing.get("doc_id", ""),
                }
            tqdm.write(f"[pipeline] {len(nodes)} chunk(s) to embed and index")

            new_doc_count = len(new_docs)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, "wb") as f:
                pickle.dump(nodes, f)
            tqdm.write(f"[pipeline] Node cache saved to {cache_path}")

        index = self._get_or_build_index()

        # Phase A: embed all nodes concurrently — one future per chunk for live progress
        texts = [n.get_content(metadata_mode=MetadataMode.NONE) for n in nodes]
        tqdm.write(
            f"[pipeline] Embedding {len(nodes)} chunk(s) via {config.EMBED_SOURCE} "
            f"(concurrency={config.EMBED_CONCURRENCY})..."
        )
        embeddings_ordered: list = [None] * len(nodes)
        embed_start = time.monotonic()
        completed = 0
        failed = 0
        with tqdm(total=len(nodes), desc="Embedding chunks", unit="chunk") as pbar:
            with ThreadPoolExecutor(max_workers=config.EMBED_CONCURRENCY) as executor:
                future_to_idx = {
                    executor.submit(self.embed_model.get_text_embedding, text): idx
                    for idx, text in enumerate(texts)
                }
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    try:
                        embeddings_ordered[idx] = future.result()
                    except Exception as exc:
                        tqdm.write(f"[pipeline] Warning: chunk {idx} embed failed ({exc}), skipping")
                        failed += 1
                    completed += 1
                    avg = completed / (time.monotonic() - embed_start)
                    pbar.set_postfix(avg=f"{avg:.2f} chunk/s", failed=failed, refresh=False)
                    pbar.update(1)

        if failed:
            tqdm.write(f"[pipeline] {failed} chunk(s) skipped due to embed errors")

        valid = [(n, e) for n, e in zip(nodes, embeddings_ordered) if e is not None]
        for node, emb in valid:
            node.embedding = emb

        # Phase B: single bulk insert — ChromaDB builds HNSW index exactly once
        tqdm.write("[pipeline] Bulk-inserting into vector store...")
        index.insert_nodes([n for n, _ in valid])
        cache_path.unlink(missing_ok=True)

        elapsed = time.monotonic() - t_start
        tqdm.write(
            f"[pipeline] Done — {new_doc_count} new document(s) ingested in {elapsed:.1f}s"
        )
        return new_doc_count

    def query(self, question: str, top_k: int = 5) -> RAGResponse:
        """Run retrieval + generation for a question (blocking, full response)."""
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

    def query_stream(self, question: str, top_k: int = 5):
        """Streaming variant — retrieval is synchronous, generation streams token by token.

        Returns a LlamaIndex StreamingResponse with:
          .response_gen  — iterator of text tokens
          .source_nodes  — retrieved chunks (available immediately after call returns)
        """
        index = self._get_or_build_index()
        engine = index.as_query_engine(
            llm=self.llm,
            similarity_top_k=top_k,
            streaming=True,
        )
        return engine.query(question)

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
