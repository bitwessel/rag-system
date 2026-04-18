"""Central configuration for the RAG system.

All values are loaded from environment variables (optionally sourced from a
local `.env` file). This module exposes plain constants — it does not validate
credentials at import time, so CLI `--help` works without a key set. Validation
happens lazily inside `RAGPipeline` when the key is actually needed.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv

load_dotenv()

VectorBackendName = Literal["chroma", "pinecone", "weaviate"]
LLMSourceName = Literal["openrouter", "ollama"]

OPENROUTER_API_KEY: str | None = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL: str = "https://openrouter.ai/api/v1"

VECTOR_BACKEND: VectorBackendName = os.getenv("VECTOR_BACKEND", "chroma").lower()  # type: ignore[assignment]

CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")

DEFAULT_LLM_MODEL: str = os.getenv("DEFAULT_LLM_MODEL", "anthropic/claude-sonnet-4-6")
DEFAULT_EMBED_MODEL: str = os.getenv(
    "DEFAULT_EMBED_MODEL", "openai/text-embedding-3-large"
)
DEFAULT_MULTIMODAL_EMBED_MODEL: str = os.getenv(
    "DEFAULT_MULTIMODAL_EMBED_MODEL", "nvidia/llama-nemotron-embed-vl-1b-v2"
)

# Local inference via Ollama (set LLM_SOURCE=ollama and/or EMBED_SOURCE=ollama)
LLM_SOURCE: LLMSourceName = os.getenv("LLM_SOURCE", "openrouter").lower()  # type: ignore[assignment]
EMBED_SOURCE: LLMSourceName = os.getenv("EMBED_SOURCE", "openrouter").lower()  # type: ignore[assignment]
OLLAMA_BASE_URL: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_LLM_MODEL: str = os.getenv("OLLAMA_LLM_MODEL", "llama3.2")
OLLAMA_EMBED_MODEL: str = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")

CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "64"))

# Ingestion tuning — lower INGEST_BATCH_SIZE or raise EMBED_BATCH_DELAY when hitting rate limits
INGEST_BATCH_SIZE: int = int(os.getenv("INGEST_BATCH_SIZE", "50"))
EMBED_RETRY_MAX: int = int(os.getenv("EMBED_RETRY_MAX", "5"))
EMBED_RETRY_BASE_DELAY: float = float(os.getenv("EMBED_RETRY_BASE_DELAY", "2.0"))
EMBED_BATCH_DELAY: float = float(os.getenv("EMBED_BATCH_DELAY", "0.0"))
EMBED_CONCURRENCY: int = int(os.getenv("EMBED_CONCURRENCY", "4"))
NODE_CACHE_PATH: str = os.getenv("NODE_CACHE_PATH", "./data/node_cache.pkl")

HTTP_REFERER: str = os.getenv(
    "OPENROUTER_HTTP_REFERER", "https://github.com/BitWessel/rag-system"
)
X_TITLE: str = os.getenv("OPENROUTER_X_TITLE", "rag-system")

IMAP_HOST: str | None = os.getenv("IMAP_HOST") or None
IMAP_USER: str | None = os.getenv("IMAP_USER") or None
IMAP_PASSWORD: str | None = os.getenv("IMAP_PASSWORD") or None

PINECONE_API_KEY: str | None = os.getenv("PINECONE_API_KEY") or None
PINECONE_ENVIRONMENT: str | None = os.getenv("PINECONE_ENVIRONMENT") or None
PINECONE_INDEX: str | None = os.getenv("PINECONE_INDEX") or None

WEAVIATE_URL: str | None = os.getenv("WEAVIATE_URL") or None
WEAVIATE_API_KEY: str | None = os.getenv("WEAVIATE_API_KEY") or None


class ConfigError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


def require_openrouter_key() -> str:
    """Return the OpenRouter API key or raise a user-friendly ConfigError."""
    if not OPENROUTER_API_KEY:
        raise ConfigError(
            "OPENROUTER_API_KEY is not set. Copy .env.example to .env and fill in "
            "your OpenRouter key (https://openrouter.ai/keys). "
            "Alternatively, set LLM_SOURCE=ollama and EMBED_SOURCE=ollama to run fully locally."
        )
    return OPENROUTER_API_KEY


def ensure_data_dirs() -> None:
    """Create local data directories used by the active backend, if needed."""
    if VECTOR_BACKEND == "chroma":
        Path(CHROMA_PERSIST_DIR).mkdir(parents=True, exist_ok=True)
