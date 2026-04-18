# RAG System

A generic, plugin-based Retrieval-Augmented Generation (RAG) system. New data
sources (email, PDFs, legal docs, images) plug in as modules; the vector
backend is swappable between ChromaDB, Pinecone, and Weaviate via a single
config value.

## Stack

- **Orchestration**: [LlamaIndex](https://www.llamaindex.ai/) (`llama-index-core`)
- **LLM & embeddings**: [OpenRouter](https://openrouter.ai) via the `openai` SDK
    - Default LLM: `anthropic/claude-sonnet-4-6`
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

## Local Ollama setup

If you want the RAG system to keep model inference local, use Ollama for both
generation and embeddings.

1. Install [Ollama](https://ollama.com/).
2. Pull the models used by this repo:

```bash
ollama pull llama3.2
ollama pull nomic-embed-text
```

3. Set your `.env` like this:

```env
LLM_SOURCE=ollama
EMBED_SOURCE=ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_LLM_MODEL=llama3.2
OLLAMA_EMBED_MODEL=nomic-embed-text
```

4. Start Ollama and confirm it is reachable on `http://localhost:11434`.

With that setup, email ingestion still connects to your mail provider over IMAP,
but chunking, embeddings, and answer generation run locally through Ollama.

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

## Gmail IMAP credentials

This project currently uses basic IMAP login in [sources/email.py](sources/email.py),
not OAuth.

For a personal Gmail account:

1. Turn on 2-Step Verification for your Google account.
2. Create an App Password for the mail client connection.
3. Use these values with `python main.py ingest email --imap`:
   - `--host imap.gmail.com`
   - `--user youraddress@gmail.com`
   - `--password <16-character app password>`

Example:

```bash
python main.py ingest email --imap --host imap.gmail.com --user you@gmail.com
```

You can also store these in `.env`:

```env
IMAP_HOST=imap.gmail.com
IMAP_USER=you@gmail.com
IMAP_PASSWORD=your_16_char_app_password
```

Notes:

- For personal Gmail accounts, Google's help docs say IMAP is always on as of
  January 2025, so you usually do not need to manually enable IMAP anymore.
- App passwords require 2-Step Verification to be enabled on the Google account.
- Google says app passwords are less preferred than OAuth, but this repo's
  current IMAP integration expects IMAP username/password style auth.
- For Google Workspace accounts, Google moved third-party mail access to
  OAuth-only on May 1, 2025. That means this repo's current IMAP password flow
  may not work for Workspace mailboxes without adding OAuth support first.

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
| `LLM_SOURCE` / `EMBED_SOURCE` | Model provider: `ollama` or `openrouter` | `openrouter` |
| `OLLAMA_BASE_URL` | Local Ollama endpoint | `http://localhost:11434` |
| `OLLAMA_LLM_MODEL` | Local generation model | `llama3.2` |
| `OLLAMA_EMBED_MODEL` | Local embedding model | `nomic-embed-text` |
| `DEFAULT_LLM_MODEL` | Generation model | `anthropic/claude-sonnet-4-6` |
| `DEFAULT_EMBED_MODEL` | Embedding model | `openai/text-embedding-3-large` |
| `IMAP_HOST` / `IMAP_USER` / `IMAP_PASSWORD` | IMAP defaults for email ingestion | — |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | LlamaIndex sentence splitter | `512` / `64` |
