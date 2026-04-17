"""Image source — stub. Wire up multimodal embedding to activate.

To finish this source:

1. In `core/embeddings.py`, add a second class (e.g. `OpenRouterMultimodalEmbedding`)
   that calls the multimodal embedding endpoint for
   `config.DEFAULT_MULTIMODAL_EMBED_MODEL` (base64-encoded images).
2. In `core/pipeline.py`, allow `RAGPipeline` to select the embedder based on
   `source.collection_name == "images"`.
3. Replace the `NotImplementedError` below with a loader that walks a
   directory of images, builds one `Document` per image with metadata
   (`file_path`, `width`, `height`, `taken_at` from EXIF if available), and
   stores the raw image bytes or path in metadata so retrieval can return
   the image.
"""
from __future__ import annotations

from typing import Any

from llama_index.core import Document


_SETUP_MSG = (
    "ImageSource is a stub. See sources/images.py for the three steps needed "
    "to wire up multimodal embedding + image ingestion."
)


class ImageSource:
    """Stubbed image source — raises on use with setup instructions."""

    collection_name: str = "images"

    def load(self, **_: Any) -> list[Document]:
        raise NotImplementedError(_SETUP_MSG)
