"""Vector backend implementations. Pick one via `VECTOR_BACKEND` in config."""
from core.backends.base import VectorBackend
from core.backends.chroma import ChromaBackend

__all__ = ["VectorBackend", "ChromaBackend", "build_backend"]


def build_backend(name: str) -> VectorBackend:
    """Factory: return the configured vector backend, or raise on unknown name."""
    name = name.lower()
    if name == "chroma":
        return ChromaBackend()
    if name == "pinecone":
        from core.backends.pinecone import PineconeBackend

        return PineconeBackend()
    if name == "weaviate":
        from core.backends.weaviate import WeaviateBackend

        return WeaviateBackend()
    raise ValueError(
        f"Unknown VECTOR_BACKEND={name!r}. Expected one of: chroma, pinecone, weaviate."
    )
