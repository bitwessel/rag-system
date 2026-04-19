"""Pluggable data sources. New types (email, PDFs, images, ...) live here.

Each source implements the `DataSource` Protocol: a `load()` method that
returns a list of LlamaIndex `Document`s with a stable `doc_id` and useful
metadata. The pipeline handles chunking, embedding, and storage.
"""
from sources.base import DataSource
from sources.email import EmailSource
from sources.images import ImageSource
from sources.paulgraham import PaulGrahamSource
from sources.text import TextSource

__all__ = ["DataSource", "EmailSource", "ImageSource", "PaulGrahamSource", "TextSource"]
