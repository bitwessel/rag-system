"""FastAPI server that serves the Nexus UI and streams RAG query results via SSE."""
from __future__ import annotations

import asyncio
import json
import queue
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, StreamingResponse
from pydantic import BaseModel

import config

app = FastAPI(title="Nexus RAG")

STATIC_DIR = Path(__file__).parent / "static"
STATS_PATH = Path("./data/model_stats.json")


def _load_model_stats() -> dict:
    try:
        return json.loads(STATS_PATH.read_text()) if STATS_PATH.exists() else {}
    except Exception:
        return {}


def _update_model_stats(model: str, ttft_ms: float) -> None:
    stats = _load_model_stats()
    s = stats.get(model, {"runs": 0, "avg_ttft_ms": 0.0})
    runs = s["runs"] + 1
    avg = (s["avg_ttft_ms"] * s["runs"] + ttft_ms) / runs
    stats[model] = {"runs": runs, "avg_ttft_ms": round(avg)}
    try:
        STATS_PATH.parent.mkdir(parents=True, exist_ok=True)
        STATS_PATH.write_text(json.dumps(stats))
    except Exception:
        pass


class QueryRequest(BaseModel):
    question: str
    collection: str = "emails"
    top_k: int = 5
    model: str | None = None


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


@app.get("/api/ollama/models")
async def list_ollama_models() -> dict:
    import urllib.request
    stats = _load_model_stats()
    try:
        url = f"{config.OLLAMA_BASE_URL}/api/tags"
        with urllib.request.urlopen(url, timeout=3) as resp:
            data = json.loads(resp.read())
        models = []
        for m in data.get("models", []):
            name = m.get("name", "")
            size_bytes = m.get("size", 0)
            s = stats.get(name, {})
            models.append({
                "name": name,
                "size_gb": round(size_bytes / 1e9, 1),
                "avg_ttft_ms": s.get("avg_ttft_ms"),
                "runs": s.get("runs", 0),
            })
        return {"models": models, "active": config.OLLAMA_LLM_MODEL}
    except Exception:
        return {"models": [], "active": config.OLLAMA_LLM_MODEL}


@app.post("/api/query/stream")
async def query_stream(req: QueryRequest) -> StreamingResponse:
    async def event_generator():
        t_start = time.monotonic()
        first_token_seen = False
        active_model = req.model or (config.OLLAMA_LLM_MODEL if config.LLM_SOURCE == "ollama" else None)

        try:
            def build_and_query():
                from core.pipeline import RAGPipeline
                source = _get_source(req.collection)
                pipeline = RAGPipeline(
                    source,
                    collection_name=req.collection,
                    llm_model_override=req.model,
                )
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
                if not first_token_seen and active_model and tok.strip():
                    ttft_ms = (time.monotonic() - t_start) * 1000
                    threading.Thread(
                        target=_update_model_stats,
                        args=(active_model, ttft_ms),
                        daemon=True,
                    ).start()
                    first_token_seen = True
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
