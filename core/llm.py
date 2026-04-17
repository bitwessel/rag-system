"""OpenRouter LLM wrapper for LlamaIndex (CustomLLM)."""
from __future__ import annotations

from typing import Any, Iterator

from llama_index.core.base.llms.types import (
    ChatMessage,
    ChatResponse,
    ChatResponseGen,
    CompletionResponse,
    CompletionResponseGen,
    LLMMetadata,
    MessageRole,
)
from llama_index.core.llms.callbacks import (
    llm_chat_callback,
    llm_completion_callback,
)
from llama_index.core.llms.custom import CustomLLM
from openai import OpenAI, OpenAIError
from pydantic import Field, PrivateAttr

import config


_CONTEXT_WINDOWS: dict[str, int] = {
    "anthropic/claude-opus-4-7": 200_000,
    "anthropic/claude-sonnet-4-6": 200_000,
    "anthropic/claude-haiku-4-5": 200_000,
}


class OpenRouterLLM(CustomLLM):
    """LlamaIndex CustomLLM that calls OpenRouter's OpenAI-compatible /chat API."""

    model: str = Field(default_factory=lambda: config.DEFAULT_LLM_MODEL)
    temperature: float = Field(default=0.1)
    max_tokens: int = Field(default=1024)
    context_window: int = Field(default=200_000)

    _client: OpenAI = PrivateAttr()

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
        base_url: str | None = None,
        temperature: float = 0.1,
        max_tokens: int = 1024,
        **kwargs: Any,
    ) -> None:
        resolved_model = model or config.DEFAULT_LLM_MODEL
        super().__init__(
            model=resolved_model,
            temperature=temperature,
            max_tokens=max_tokens,
            context_window=_CONTEXT_WINDOWS.get(resolved_model, 128_000),
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
        return "OpenRouterLLM"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(
            context_window=self.context_window,
            num_output=self.max_tokens,
            model_name=self.model,
            is_chat_model=True,
        )

    @staticmethod
    def _to_openai_messages(messages: list[ChatMessage]) -> list[dict[str, str]]:
        return [{"role": m.role.value, "content": m.content or ""} for m in messages]

    @llm_completion_callback()
    def complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponse:
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except OpenAIError as exc:
            raise RuntimeError(
                f"OpenRouter LLM call failed for model {self.model!r}: {exc}."
            ) from exc
        text = resp.choices[0].message.content or ""
        return CompletionResponse(text=text, raw=resp.model_dump())

    @llm_completion_callback()
    def stream_complete(
        self, prompt: str, formatted: bool = False, **kwargs: Any
    ) -> CompletionResponseGen:
        try:
            stream = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
            )
        except OpenAIError as exc:
            raise RuntimeError(
                f"OpenRouter streaming call failed for model {self.model!r}: {exc}."
            ) from exc

        def gen() -> Iterator[CompletionResponse]:
            acc = ""
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content or ""
                if not delta:
                    continue
                acc += delta
                yield CompletionResponse(text=acc, delta=delta)

        return gen()

    @llm_chat_callback()
    def chat(self, messages: list[ChatMessage], **kwargs: Any) -> ChatResponse:
        try:
            resp = self._client.chat.completions.create(
                model=self.model,
                messages=self._to_openai_messages(messages),
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        except OpenAIError as exc:
            raise RuntimeError(
                f"OpenRouter chat call failed for model {self.model!r}: {exc}."
            ) from exc
        text = resp.choices[0].message.content or ""
        return ChatResponse(
            message=ChatMessage(role=MessageRole.ASSISTANT, content=text),
            raw=resp.model_dump(),
        )

    @llm_chat_callback()
    def stream_chat(
        self, messages: list[ChatMessage], **kwargs: Any
    ) -> ChatResponseGen:
        try:
            stream = self._client.chat.completions.create(
                model=self.model,
                messages=self._to_openai_messages(messages),
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                stream=True,
            )
        except OpenAIError as exc:
            raise RuntimeError(
                f"OpenRouter streaming chat failed for model {self.model!r}: {exc}."
            ) from exc

        def gen() -> Iterator[ChatResponse]:
            acc = ""
            for chunk in stream:
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta.content or ""
                if not delta:
                    continue
                acc += delta
                yield ChatResponse(
                    message=ChatMessage(role=MessageRole.ASSISTANT, content=acc),
                    delta=delta,
                )

        return gen()
