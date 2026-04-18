"""OpenRouter embedding wrapper for LlamaIndex.

Uses the `openai` SDK pointed at `https://openrouter.ai/api/v1`. OpenRouter's
embedding coverage is limited; if the configured model is not routable,
OpenRouter returns a 4xx and we surface a clear error pointing the user at
`DEFAULT_EMBED_MODEL` in `.env`.
"""
from __future__ import annotations

import time
from typing import Any

from llama_index.core.embeddings import BaseEmbedding
from openai import APIStatusError, OpenAI, OpenAIError, RateLimitError
from pydantic import Field, PrivateAttr
from tqdm import tqdm

import config


class OpenRouterEmbedding(BaseEmbedding):
    """LlamaIndex embedding backed by OpenRouter's OpenAI-compatible API."""

    model_name: str = Field(default_factory=lambda: config.DEFAULT_EMBED_MODEL)

    _client: OpenAI = PrivateAttr()

    def __init__(
        self,
        model_name: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        embed_batch_size: int = 64,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_name=model_name or config.DEFAULT_EMBED_MODEL,
            embed_batch_size=embed_batch_size,
            **kwargs,
        )
        self._client = OpenAI(
            api_key=api_key or config.require_openrouter_key(),
            base_url=base_url or config.OPENROUTER_BASE_URL,
            default_headers={
                "HTTP-Referer": config.HTTP_REFERER,
                "X-Title": config.X_TITLE,
            },
        )

    @classmethod
    def class_name(cls) -> str:
        return "OpenRouterEmbedding"

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        max_retries = config.EMBED_RETRY_MAX
        base_delay = config.EMBED_RETRY_BASE_DELAY
        for attempt in range(max_retries + 1):
            try:
                resp = self._client.embeddings.create(
                    model=self.model_name,
                    input=texts,
                )
                return [d.embedding for d in resp.data]
            except RateLimitError as exc:
                if attempt == max_retries:
                    raise RuntimeError(
                        f"Embedding rate limit hit — gave up after {max_retries} retries. "
                        "Try lowering INGEST_BATCH_SIZE or adding EMBED_BATCH_DELAY in .env."
                    ) from exc
                wait = base_delay * (2**attempt)
                tqdm.write(
                    f"[embed] Rate limit hit (attempt {attempt + 1}/{max_retries}), "
                    f"retrying in {wait:.1f}s..."
                )
                time.sleep(wait)
            except APIStatusError as exc:
                if exc.status_code >= 500 and attempt < max_retries:
                    wait = base_delay * (2**attempt)
                    tqdm.write(
                        f"[embed] Server error {exc.status_code} (attempt {attempt + 1}/{max_retries}), "
                        f"retrying in {wait:.1f}s..."
                    )
                    time.sleep(wait)
                    continue
                raise RuntimeError(
                    f"OpenRouter embedding call failed for model {self.model_name!r}: "
                    f"{exc}. If the model is not supported by OpenRouter's /embeddings "
                    "endpoint, change DEFAULT_EMBED_MODEL in .env to a supported model."
                ) from exc
            except OpenAIError as exc:
                raise RuntimeError(
                    f"OpenRouter embedding call failed for model {self.model_name!r}: "
                    f"{exc}. If the model is not supported by OpenRouter's /embeddings "
                    "endpoint, change DEFAULT_EMBED_MODEL in .env to a supported model."
                ) from exc
        return []  # unreachable, satisfies type checker

    def _get_text_embedding(self, text: str) -> list[float]:
        return self._embed([text])[0]

    def _get_text_embeddings(self, texts: list[str]) -> list[list[float]]:
        return self._embed(texts)

    def _get_query_embedding(self, query: str) -> list[float]:
        return self._embed([query])[0]

    async def _aget_text_embedding(self, text: str) -> list[float]:
        return self._get_text_embedding(text)

    async def _aget_query_embedding(self, query: str) -> list[float]:
        return self._get_query_embedding(query)
