"""Paul Graham essays source: scrapes paulgraham.com and caches locally."""
from __future__ import annotations

import hashlib
import re
import time
import urllib.request
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup
from llama_index.core import Document
from tqdm import tqdm


class PaulGrahamSource:
    collection_name: str = "paulgraham"

    _BASE = "https://paulgraham.com"
    _INDEX = "https://paulgraham.com/articles.html"
    _CACHE_DIR = Path("./data/paulgraham")
    _DELAY = 0.5  # polite crawl delay

    def load(
        self,
        limit: int | None = None,
        refresh: bool = False,
        **_: Any,
    ) -> list[Document]:
        self._CACHE_DIR.mkdir(parents=True, exist_ok=True)
        urls = self._fetch_essay_urls()
        if limit:
            urls = urls[:limit]

        docs: list[Document] = []
        for url in tqdm(urls, desc="Paul Graham essays", unit="essay"):
            slug = url.rstrip("/").rsplit("/", 1)[-1].replace(".html", "")
            cache_file = self._CACHE_DIR / f"{slug}.txt"

            if cache_file.exists() and not refresh:
                raw = cache_file.read_text(encoding="utf-8")
                lines = raw.split("\n", 2)
                title = lines[0].strip()
                text = lines[2].strip() if len(lines) > 2 else raw
            else:
                try:
                    title, text = self._fetch_essay(url)
                    cache_file.write_text(f"{title}\n\n{text}", encoding="utf-8")
                    time.sleep(self._DELAY)
                except Exception as exc:
                    tqdm.write(f"[paulgraham] Skipping {url}: {exc}")
                    continue

            if not text.strip():
                continue

            doc_id = hashlib.sha1(url.encode()).hexdigest()
            docs.append(
                Document(
                    text=f"Title: {title}\n\n{text}",
                    doc_id=doc_id,
                    metadata={
                        "source_type": "paulgraham",
                        "url": url,
                        "slug": slug,
                        "title": title,
                    },
                )
            )
        return docs

    def _fetch_essay_urls(self) -> list[str]:
        html = self._get(self._INDEX)
        soup = BeautifulSoup(html, "html.parser")
        urls: list[str] = []
        seen: set[str] = set()
        for a in soup.find_all("a", href=True):
            href: str = a["href"]
            # Only local essay pages — no slashes, ends in .html
            if not href.endswith(".html") or "/" in href:
                continue
            full = f"{self._BASE}/{href}"
            if full not in seen:
                seen.add(full)
                urls.append(full)
        return urls

    def _fetch_essay(self, url: str) -> tuple[str, str]:
        html = self._get(url)
        soup = BeautifulSoup(html, "html.parser")

        title = ""
        title_tag = soup.find("title")
        if title_tag:
            title = title_tag.get_text().strip()

        for tag in soup(["script", "style", "head"]):
            tag.decompose()

        text = soup.get_text(separator="\n")
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        return title, text

    @staticmethod
    def _get(url: str) -> str:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.read().decode("utf-8", errors="replace")
