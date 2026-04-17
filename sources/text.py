"""Text/document source: .txt, .md, .pdf from a file or directory."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from llama_index.core import Document, SimpleDirectoryReader


class TextSource:
    """Loads plain text, Markdown, and PDF files as Documents."""

    collection_name: str = "documents"

    SUPPORTED_EXTS: tuple[str, ...] = (".txt", ".md", ".markdown", ".pdf")

    def load(self, path: str | None = None, **_: Any) -> list[Document]:
        if not path:
            raise ValueError("TextSource.load requires `path=...` (file or directory).")
        target = Path(path)
        if not target.exists():
            raise FileNotFoundError(f"Path not found: {path}")

        if target.is_file():
            files = [target]
        else:
            files = [
                p
                for p in target.rglob("*")
                if p.is_file() and p.suffix.lower() in self.SUPPORTED_EXTS
            ]
        if not files:
            return []

        reader = SimpleDirectoryReader(
            input_files=[str(f) for f in files],
            required_exts=list(self.SUPPORTED_EXTS),
        )
        raw_docs = reader.load_data()

        out: list[Document] = []
        for d in raw_docs:
            file_path = d.metadata.get("file_path") or d.metadata.get("filename") or ""
            page = d.metadata.get("page_label") or d.metadata.get("page", "")
            doc_id = self._doc_id(file_path, page, d.text)
            out.append(
                Document(
                    text=d.text,
                    doc_id=doc_id,
                    metadata={
                        "source_type": "document",
                        "file_path": file_path,
                        "page": str(page) if page else "",
                    },
                )
            )
        return out

    @staticmethod
    def _doc_id(file_path: str, page: str, text: str) -> str:
        basis = f"{file_path}|{page}|{hashlib.sha1(text.encode('utf-8')).hexdigest()}"
        return hashlib.sha1(basis.encode("utf-8")).hexdigest()
