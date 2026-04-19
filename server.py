"""FastAPI server that serves the Nexus UI and streams RAG query results via SSE."""
from __future__ import annotations

import asyncio
import json
import queue
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

import config

app = FastAPI(title="Nexus RAG")

STATIC_DIR = Path(__file__).parent / "static"


class QueryRequest(BaseModel):
    question: str
    collection: str = "emails"
    top_k: int = 5


def _get_source(collection: str):
    from sources.email import EmailSource
    from sources.text import TextSource
    return EmailSource() if collection.lower() in ("emails", "email") else TextSource()


def _extract_sources(response) -> list[dict[str, Any]]:
    sources = []
    for node in getattr(response, "source_nodes", []) or []:
        meta = node.metadata or {}
        score = getattr(node, "score", None)
        src: dict[str, Any] = {"score": float(score) if score is not None else 0.0}
        for k, v in meta.items():
            if isinstance(v, (str, int, float, bool)):
                src[k] = v
        if hasattr(node, "get_content"):
            src["text"] = node.get_content()[:400]
        sources.append(src)
    return sources


@app.get("/")
async def index() -> HTMLResponse:
    html = (STATIC_DIR / "nexus.html").read_text(encoding="utf-8")
    return HTMLResponse(html)


@app.get("/api/collections")
async def list_collections() -> dict:
    try:
        from core.backends.chroma import ChromaBackend
        backend = ChromaBackend()
        names = [c.name for c in backend._client.list_collections()]
        return {"collections": names or []}
    except Exception:
        return {"collections": []}


@app.post("/api/query/stream")
async def query_stream(req: QueryRequest) -> StreamingResponse:
    async def event_generator():
        try:
            def build_and_query():
                from core.pipeline import RAGPipeline
                source = _get_source(req.collection)
                pipeline = RAGPipeline(source, collection_name=req.collection)
                return pipeline.query_stream(req.question, top_k=req.top_k)

            response = await asyncio.to_thread(build_and_query)

            sources = _extract_sources(response)
            yield f"data: {json.dumps({'type': 'sources', 'sources': sources})}\n\n"

            token_q: queue.Queue[str | None] = queue.Queue()

            def drain() -> None:
                try:
                    for tok in response.response_gen:
                        token_q.put(tok)
                finally:
                    token_q.put(None)

            threading.Thread(target=drain, daemon=True).start()

            loop = asyncio.get_event_loop()
            while True:
                tok = await loop.run_in_executor(None, token_q.get)
                if tok is None:
                    break
                yield f"data: {json.dumps({'type': 'token', 'text': tok})}\n\n"

            yield f"data: {json.dumps({'type': 'done'})}\n\n"

        except Exception as exc:
            yield f"data: {json.dumps({'type': 'error', 'message': str(exc)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
