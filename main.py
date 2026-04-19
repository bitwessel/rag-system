"""CLI for the RAG system.

Examples:

    python main.py ingest email --mbox path/to/mail.mbox
    python main.py ingest email --imap --host imap.gmail.com --user you@gmail.com
    python main.py ingest text --path ./docs
    python main.py query "What did John say about the deadline?"
    python main.py query "Find all emails about invoice payments" --top-k 10
"""
from __future__ import annotations

import sys
import traceback
from typing import Any

import click

import config
from config import ConfigError

# stdout encoding fix for Windows terminals that default to cp1252
if sys.stdout.encoding and sys.stdout.encoding.lower().startswith("cp"):
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def _die(msg: str, *, debug: bool = False) -> None:
    click.echo(f"Error: {msg}", err=True)
    if debug:
        traceback.print_exc()
    sys.exit(1)


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.option("--debug", is_flag=True, help="Show full tracebacks on error.")
@click.pass_context
def cli(ctx: click.Context, debug: bool) -> None:
    """Generic RAG system: ingest data sources and query them."""
    ctx.ensure_object(dict)
    ctx.obj["debug"] = debug


# --------------------------------------------------------------------------- #
# ingest                                                                      #
# --------------------------------------------------------------------------- #


@cli.group()
def ingest() -> None:
    """Ingest a data source into the vector store."""


@ingest.command("email")
@click.option("--mbox", type=click.Path(exists=True, dir_okay=False), default=None,
              help="Path to an .mbox file.")
@click.option("--imap", "use_imap", is_flag=True, help="Use IMAP instead of mbox.")
@click.option("--host", default=None, help="IMAP host (e.g. imap.gmail.com). "
              "Falls back to IMAP_HOST from .env.")
@click.option("--user", default=None, help="IMAP username. Falls back to IMAP_USER.")
@click.option("--password", default=None, help="IMAP password (app password for "
              "Gmail). Falls back to IMAP_PASSWORD. Prompted if missing.")
@click.option("--folder", "folders", multiple=True,
              help="IMAP folder to pull from. Repeat to add more. "
                   "Defaults to INBOX + Sent.")
@click.option("--limit", type=int, default=None,
              help="Max messages per folder (newest first). Omit for all.")
@click.option("--collection", default=None,
              help="Override the ChromaDB collection name. Default: 'emails'.")
@click.option("--fresh", is_flag=True,
              help="Drop the existing collection before ingesting (full re-index).")
@click.option("--list-folders", "list_folders_only", is_flag=True,
              help="Print available IMAP folders and exit (no ingestion).")
@click.option("--save-mbox", default=None, type=click.Path(),
              help="Save all fetched IMAP messages to this .mbox file. "
                   "Useful for resuming without re-fetching if embedding fails.")
@click.pass_context
def ingest_email(
    ctx: click.Context,
    mbox: str | None,
    use_imap: bool,
    host: str | None,
    user: str | None,
    password: str | None,
    folders: tuple[str, ...],
    limit: int | None,
    collection: str | None,
    fresh: bool,
    list_folders_only: bool,
    save_mbox: str | None,
) -> None:
    """Ingest emails from an mbox file or an IMAP server."""
    debug = ctx.obj.get("debug", False)
    if list_folders_only and not use_imap:
        _die("--list-folders requires --imap.", debug=debug)
    if not list_folders_only and bool(mbox) == bool(use_imap):
        _die("Specify exactly one of --mbox PATH or --imap.", debug=debug)

    try:
        from core.pipeline import RAGPipeline
        from sources.email import EmailSource

        source = EmailSource()

        if list_folders_only:
            resolved_host = host or config.IMAP_HOST
            resolved_user = user or config.IMAP_USER
            resolved_pw = password or config.IMAP_PASSWORD
            if not resolved_host:
                _die("IMAP host not provided. Pass --host or set IMAP_HOST in .env.", debug=debug)
            if not resolved_user:
                _die("IMAP user not provided. Pass --user or set IMAP_USER in .env.", debug=debug)
            if not resolved_pw:
                resolved_pw = click.prompt("IMAP password", hide_input=True)
            available = source.list_imap_folders(resolved_host, resolved_user, resolved_pw)
            click.echo("Available IMAP folders:")
            for f in available:
                click.echo(f"  {f}")
            return

        pipeline = RAGPipeline(source, collection_name=collection)

        if mbox:
            count = pipeline.ingest(mbox=mbox, fresh=fresh)
        else:
            resolved_host = host or config.IMAP_HOST
            resolved_user = user or config.IMAP_USER
            resolved_pw = password or config.IMAP_PASSWORD
            if not resolved_host:
                _die("IMAP host not provided. Pass --host or set IMAP_HOST in .env.",
                     debug=debug)
            if not resolved_user:
                _die("IMAP user not provided. Pass --user or set IMAP_USER in .env.",
                     debug=debug)
            if not resolved_pw:
                resolved_pw = click.prompt("IMAP password", hide_input=True)
            imap_kwargs: dict[str, Any] = {
                "host": resolved_host,
                "user": resolved_user,
                "password": resolved_pw,
            }
            if folders:
                imap_kwargs["folders"] = list(folders)
            if limit is not None:
                imap_kwargs["limit"] = limit
            if save_mbox:
                imap_kwargs["save_mbox"] = save_mbox
            count = pipeline.ingest(imap=imap_kwargs, fresh=fresh)

        click.echo(f"Ingested {count} new email(s) into '{pipeline.collection_name}'.")

    except ConfigError as exc:
        _die(str(exc), debug=debug)
    except NotImplementedError as exc:
        _die(str(exc), debug=debug)
    except Exception as exc:  # noqa: BLE001
        _die(f"{type(exc).__name__}: {exc}", debug=debug)


@ingest.command("text")
@click.option("--path", required=True, type=click.Path(exists=True),
              help="File or directory of .txt/.md/.pdf documents.")
@click.option("--collection", default=None,
              help="Override the ChromaDB collection name. Default: 'documents'.")
@click.option("--fresh", is_flag=True,
              help="Drop the existing collection before ingesting (full re-index).")
@click.pass_context
def ingest_text(ctx: click.Context, path: str, collection: str | None, fresh: bool) -> None:
    """Ingest .txt / .md / .pdf documents from a file or directory."""
    debug = ctx.obj.get("debug", False)
    try:
        from core.pipeline import RAGPipeline
        from sources.text import TextSource

        pipeline = RAGPipeline(TextSource(), collection_name=collection)
        count = pipeline.ingest(path=path, fresh=fresh)
        click.echo(
            f"Ingested {count} new document chunk-source(s) into "
            f"'{pipeline.collection_name}'."
        )
    except ConfigError as exc:
        _die(str(exc), debug=debug)
    except NotImplementedError as exc:
        _die(str(exc), debug=debug)
    except Exception as exc:  # noqa: BLE001
        _die(f"{type(exc).__name__}: {exc}", debug=debug)


@ingest.command("paulgraham")
@click.option("--limit", type=int, default=None,
              help="Max essays to fetch (first N from index). Omit for all ~220.")
@click.option("--collection", default=None,
              help="Override ChromaDB collection name. Default: 'paulgraham'.")
@click.option("--fresh", is_flag=True,
              help="Drop existing collection before ingesting.")
@click.option("--refresh", is_flag=True,
              help="Re-download essays even if already cached in ./data/paulgraham/.")
@click.pass_context
def ingest_paulgraham(
    ctx: click.Context,
    limit: int | None,
    collection: str | None,
    fresh: bool,
    refresh: bool,
) -> None:
    """Ingest Paul Graham essays scraped from paulgraham.com (cached locally)."""
    debug = ctx.obj.get("debug", False)
    try:
        from core.pipeline import RAGPipeline
        from sources.paulgraham import PaulGrahamSource

        pipeline = RAGPipeline(PaulGrahamSource(), collection_name=collection)
        count = pipeline.ingest(limit=limit, refresh=refresh, fresh=fresh)
        click.echo(f"Ingested {count} new essay(s) into '{pipeline.collection_name}'.")
    except ConfigError as exc:
        _die(str(exc), debug=debug)
    except NotImplementedError as exc:
        _die(str(exc), debug=debug)
    except Exception as exc:  # noqa: BLE001
        _die(f"{type(exc).__name__}: {exc}", debug=debug)


# --------------------------------------------------------------------------- #
# query                                                                       #
# --------------------------------------------------------------------------- #


@cli.command("query")
@click.argument("question")
@click.option("--top-k", type=int, default=5, help="How many chunks to retrieve.")
@click.option("--collection", default="emails",
              help="Which ChromaDB collection to search. Default: 'emails'.")
@click.option("--show-sources/--no-sources", default=True,
              help="Print source metadata alongside the answer.")
@click.pass_context
def query_cmd(
    ctx: click.Context,
    question: str,
    top_k: int,
    collection: str,
    show_sources: bool,
) -> None:
    """Ask a question against a previously-ingested collection."""
    debug = ctx.obj.get("debug", False)
    try:
        from core.pipeline import RAGPipeline
        from sources.email import EmailSource
        from sources.text import TextSource

        from sources.paulgraham import PaulGrahamSource
        if collection.lower() in ("emails", "email"):
            source = EmailSource()
        elif collection.lower() in ("paulgraham", "paul_graham", "pg"):
            source = PaulGrahamSource()
        else:
            source = TextSource()
        pipeline = RAGPipeline(source, collection_name=collection)

        llm_label = (
            f"{config.OLLAMA_LLM_MODEL} via ollama"
            if config.LLM_SOURCE == "ollama"
            else config.DEFAULT_LLM_MODEL
        )
        click.echo(
            f"[query] Retrieving top-{top_k} chunk(s) from '{collection}'...",
            err=True,
        )
        response = pipeline.query_stream(question, top_k=top_k)

        # source_nodes are available immediately after retrieval (before generation)
        source_nodes = getattr(response, "source_nodes", []) or []
        click.echo(
            f"[query] Generating answer with {llm_label}...", err=True
        )

        click.echo("\nAnswer:\n")
        answer_text = ""
        for token in response.response_gen:
            click.echo(token, nl=False)
            sys.stdout.flush()
            answer_text += token
        click.echo()  # final newline

        if not answer_text.strip():
            click.echo("(no answer generated)")

        if show_sources and source_nodes:
            click.echo("\nSources:")
            for i, node in enumerate(source_nodes, 1):
                meta = node.metadata or {}
                score = getattr(node, "score", None)
                score_s = f" (score={score:.3f})" if isinstance(score, float) else ""
                if meta.get("source_type") == "email":
                    click.echo(
                        f"  [{i}]{score_s} {meta.get('subject', '(no subject)')} "
                        f"— {meta.get('from_addr', '')} — {meta.get('date', '')}"
                    )
                else:
                    label = meta.get("file_path") or meta.get("message_id") or "unknown"
                    click.echo(f"  [{i}]{score_s} {label}")

    except ConfigError as exc:
        _die(str(exc), debug=debug)
    except NotImplementedError as exc:
        _die(str(exc), debug=debug)
    except Exception as exc:  # noqa: BLE001
        _die(f"{type(exc).__name__}: {exc}", debug=debug)


if __name__ == "__main__":
    cli(obj={})
