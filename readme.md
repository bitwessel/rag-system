# RAG System

A generic, plugin-based Retrieval-Augmented Generation (RAG) system. New data
sources (email, PDFs, legal docs, images) plug in as modules; the vector
backend is swappable between ChromaDB, Pinecone, and Weaviate via a single
config value.

## Stack

- **Orchestration**: [LlamaIndex](https://www.llamaindex.ai/) (`llama-index-core`)
- **LLM & embeddings**: [OpenRouter](https://openrouter.ai) via the `openai` SDK
    - Default LLM: `anthropic/claude-opus-4-7`
    - Default embeddings: `openai/text-embedding-3-large`
- **Vector DB**: ChromaDB (active), Pinecone / Weaviate (stubs)

## Quickstart

```bash
# 1. Install deps (use the existing venv)
venv\Scripts\activate            # Windows
# source venv/bin/activate       # macOS/Linux
pip install -r requirements.txt

# 2. Configure
copy .env.example .env           # Windows
# cp .env.example .env           # macOS/Linux
# then edit .env and set OPENROUTER_API_KEY=sk-or-...

# 3. Ingest emails from an .mbox file
python main.py ingest email --mbox path\to\mail.mbox

# 4. Ask questions
python main.py query "What did John say about the project deadline?"
python main.py query "Find all emails about invoice payments" --top-k 10
```

## Email ingestion

Two modes:

```bash
# .mbox file (e.g. exported from Thunderbird, Apple Mail, or Google Takeout)
python main.py ingest email --mbox ./mail.mbox

# Live IMAP (for Gmail, use an App Password, NOT your account password)
python main.py ingest email --imap \
  --host imap.gmail.com --user you@gmail.com
# password is prompted if --password is omitted
```

IMAP defaults to pulling `INBOX` and `[Gmail]/Sent Mail` (falls back to `Sent`).
Use `--folder NAME` (repeatable) to override, `--limit N` to cap per folder.

Dedup is by `Message-Id` — re-running `ingest` on the same mailbox is safe.

## Documents (.txt, .md, .pdf)

```bash
python main.py ingest text --path ./docs
python main.py query "summarize the contract" --collection documents
```

## Architecture

```
                 ┌────────────┐
                 │  main.py   │  click CLI
                 └─────┬──────┘
                       │
                 ┌─────▼──────┐
                 │ RAGPipeline│  core/pipeline.py
                 └──┬───┬───┬─┘
                    │   │   │
            ┌───────┘   │   └──────────┐
            │           │              │
    ┌───────▼──┐  ┌─────▼──────┐  ┌────▼──────────┐
    │DataSource│  │OpenRouter  │  │VectorBackend  │
    │(email,   │  │  LLM +     │  │(chroma active,│
    │ text,... │  │ embeddings │  │ pinecone/     │
    │  plugin) │  │            │  │ weaviate stub)│
    └──────────┘  └────────────┘  └───────────────┘
```

## Swapping the vector backend

Set `VECTOR_BACKEND` in `.env`:

- `chroma` (default, local persistent store in `./data/chroma`)
- `pinecone` — stub; follow the setup instructions printed on first use and
  in [core/backends/pinecone.py](core/backends/pinecone.py)
- `weaviate` — stub; see [core/backends/weaviate.py](core/backends/weaviate.py)

## Adding a new source

1. Create `sources/<name>.py` with a class implementing the `DataSource`
   protocol ([sources/base.py](sources/base.py)): `collection_name: str` and
   `load(**kwargs) -> list[Document]`. Set a stable `doc_id` on each Document
   so re-ingests dedupe.
2. Add it to [sources/__init__.py](sources/__init__.py).
3. Add a subcommand to [main.py](main.py) following the pattern of
   `ingest_email` / `ingest_text`.

## Known limitations

- **OpenRouter embedding coverage is limited.** If `DEFAULT_EMBED_MODEL` is
  not routable via OpenRouter's `/embeddings` endpoint, the embed call fails
  with a clear error. Change `DEFAULT_EMBED_MODEL` in `.env` to a supported
  model, or swap the embedding client in
  [core/embeddings.py](core/embeddings.py) to point at OpenAI directly.
- The image source is a stub — multimodal (`nvidia/llama-nemotron-embed-vl-1b-v2`)
  is planned for the next iteration. See [sources/images.py](sources/images.py).
- Live Gmail API ingestion (OAuth) is planned; IMAP is the interim path.

## Config reference

See [.env.example](.env.example) for the full list. Key variables:

| Variable | Purpose | Default |
|----------|---------|---------|
| `OPENROUTER_API_KEY` | OpenRouter key (required) | — |
| `VECTOR_BACKEND` | `chroma` \| `pinecone` \| `weaviate` | `chroma` |
| `CHROMA_PERSIST_DIR` | Where ChromaDB stores vectors | `./data/chroma` |
| `DEFAULT_LLM_MODEL` | Generation model | `anthropic/claude-opus-4-7` |
| `DEFAULT_EMBED_MODEL` | Embedding model | `openai/text-embedding-3-large` |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | LlamaIndex sentence splitter | `512` / `64` |
