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
) -> None:
    """Ingest emails from an mbox file or an IMAP server."""
    debug = ctx.obj.get("debug", False)
    if bool(mbox) == bool(use_imap):
        _die("Specify exactly one of --mbox PATH or --imap.", debug=debug)

    try:
        from core.pipeline import RAGPipeline
        from sources.email import EmailSource

        pipeline = RAGPipeline(EmailSource(), collection_name=collection)

        if mbox:
            count = pipeline.ingest(mbox=mbox)
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
            count = pipeline.ingest(imap=imap_kwargs)

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
@click.pass_context
def ingest_text(ctx: click.Context, path: str, collection: str | None) -> None:
    """Ingest .txt / .md / .pdf documents from a file or directory."""
    debug = ctx.obj.get("debug", False)
    try:
        from core.pipeline import RAGPipeline
        from sources.text import TextSource

        pipeline = RAGPipeline(TextSource(), collection_name=collection)
        count = pipeline.ingest(path=path)
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

        source_factory = EmailSource if collection == "emails" else TextSource
        pipeline = RAGPipeline(source_factory(), collection_name=collection)
        response = pipeline.query(question, top_k=top_k)

        click.echo("\nAnswer:\n")
        click.echo(response.answer.strip() or "(no answer generated)")

        if show_sources and response.sources:
            click.echo("\nSources:")
            for i, src in enumerate(response.sources, 1):
                score = src.get("score")
                score_s = f" (score={score:.3f})" if isinstance(score, float) else ""
                if src.get("source_type") == "email":
                    click.echo(
                        f"  [{i}]{score_s} {src.get('subject', '(no subject)')} "
                        f"— {src.get('from_addr', '')} — {src.get('date', '')}"
                    )
                else:
                    label = src.get("file_path") or src.get("message_id") or "unknown"
                    click.echo(f"  [{i}]{score_s} {label}")

    except ConfigError as exc:
        _die(str(exc), debug=debug)
    except NotImplementedError as exc:
        _die(str(exc), debug=debug)
    except Exception as exc:  # noqa: BLE001
        _die(f"{type(exc).__name__}: {exc}", debug=debug)


if __name__ == "__main__":
    cli(obj={})
