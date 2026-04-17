"""Email source: ingests from .mbox files or live IMAP (INBOX + Sent).

Produces one `Document` per message. Body is preferred `text/plain`; falls
back to stripped HTML. `doc_id` is the message's `Message-Id` header when
present, otherwise a hash of subject+date+from so re-ingests still dedupe.
"""
from __future__ import annotations

import email as stdlib_email
import hashlib
import imaplib
import mailbox
from email.header import decode_header, make_header
from email.message import Message
from email.utils import parseaddr
from typing import Any, Iterable

from bs4 import BeautifulSoup
from llama_index.core import Document

DEFAULT_IMAP_FOLDERS: tuple[str, ...] = ("INBOX", "[Gmail]/Sent Mail", "Sent")


class EmailSource:
    """Loads email into LlamaIndex Documents from an mbox file or IMAP server."""

    collection_name: str = "emails"

    def load(
        self,
        mbox: str | None = None,
        imap: dict[str, Any] | None = None,
        **_: Any,
    ) -> list[Document]:
        if mbox and imap:
            raise ValueError("Pass either `mbox` or `imap`, not both.")
        if mbox:
            return self.load_from_mbox(mbox)
        if imap:
            return self.load_from_imap(**imap)
        raise ValueError("EmailSource.load requires `mbox=...` or `imap={...}`.")

    # ----- mbox ---------------------------------------------------------------

    def load_from_mbox(self, path: str) -> list[Document]:
        box = mailbox.mbox(path)
        seen: set[str] = set()
        docs: list[Document] = []
        for msg in box:
            doc = self._build_doc(msg, folder="mbox")
            if doc.doc_id in seen:
                continue
            seen.add(doc.doc_id)
            docs.append(doc)
        return docs

    # ----- IMAP ---------------------------------------------------------------

    def load_from_imap(
        self,
        host: str,
        user: str,
        password: str,
        folders: Iterable[str] = DEFAULT_IMAP_FOLDERS,
        limit: int | None = None,
    ) -> list[Document]:
        docs: list[Document] = []
        seen: set[str] = set()
        with imaplib.IMAP4_SSL(host) as conn:
            conn.login(user, password)
            available_folders = self._list_folders(conn)
            for folder in folders:
                if folder not in available_folders:
                    continue
                typ, _ = conn.select(f'"{folder}"', readonly=True)
                if typ != "OK":
                    continue
                typ, data = conn.search(None, "ALL")
                if typ != "OK" or not data or not data[0]:
                    continue
                ids = data[0].split()
                if limit is not None:
                    ids = ids[-limit:]
                for num in ids:
                    typ, msg_data = conn.fetch(num, "(RFC822)")
                    if typ != "OK" or not msg_data:
                        continue
                    raw = next(
                        (
                            part[1]
                            for part in msg_data
                            if isinstance(part, tuple) and len(part) >= 2
                        ),
                        None,
                    )
                    if not raw:
                        continue
                    msg = stdlib_email.message_from_bytes(raw)
                    doc = self._build_doc(msg, folder=folder)
                    if doc.doc_id in seen:
                        continue
                    seen.add(doc.doc_id)
                    docs.append(doc)
            conn.logout()
        return docs

    @staticmethod
    def _list_folders(conn: imaplib.IMAP4_SSL) -> set[str]:
        typ, data = conn.list()
        if typ != "OK" or not data:
            return set()
        names: set[str] = set()
        for entry in data:
            if not entry:
                continue
            decoded = entry.decode("utf-8", errors="replace")
            # Format: (\HasNoChildren) "/" "INBOX"
            if '"' in decoded:
                names.add(decoded.rsplit('"', 2)[-2])
        return names

    # ----- shared -------------------------------------------------------------

    def _build_doc(self, msg: Message, folder: str) -> Document:
        subject = self._header(msg, "Subject")
        from_raw = self._header(msg, "From")
        to_raw = self._header(msg, "To")
        date = self._header(msg, "Date")
        message_id = self._header(msg, "Message-Id") or self._fallback_id(
            subject, from_raw, date
        )

        body = self._extract_body(msg)
        from_name, from_addr = parseaddr(from_raw)
        _, to_addr = parseaddr(to_raw)

        text = (
            f"Subject: {subject}\n"
            f"From: {from_raw}\n"
            f"To: {to_raw}\n"
            f"Date: {date}\n"
            f"Folder: {folder}\n\n"
            f"{body}"
        )
        return Document(
            text=text,
            doc_id=message_id,
            metadata={
                "source_type": "email",
                "message_id": message_id,
                "subject": subject,
                "from_addr": from_addr,
                "from_name": from_name,
                "to_addr": to_addr,
                "date": date,
                "folder": folder,
            },
        )

    @staticmethod
    def _header(msg: Message, name: str) -> str:
        raw = msg.get(name)
        if not raw:
            return ""
        try:
            return str(make_header(decode_header(raw))).strip()
        except Exception:
            return raw.strip() if isinstance(raw, str) else str(raw)

    @staticmethod
    def _fallback_id(subject: str, from_raw: str, date: str) -> str:
        h = hashlib.sha1(f"{subject}|{from_raw}|{date}".encode("utf-8")).hexdigest()
        return f"fallback-{h}"

    @staticmethod
    def _extract_body(msg: Message) -> str:
        plain_parts: list[str] = []
        html_parts: list[str] = []
        if msg.is_multipart():
            for part in msg.walk():
                ctype = part.get_content_type()
                disp = str(part.get("Content-Disposition") or "")
                if "attachment" in disp.lower():
                    continue
                if ctype == "text/plain":
                    plain_parts.append(EmailSource._decode_part(part))
                elif ctype == "text/html":
                    html_parts.append(EmailSource._decode_part(part))
        else:
            ctype = msg.get_content_type()
            if ctype == "text/plain":
                plain_parts.append(EmailSource._decode_part(msg))
            elif ctype == "text/html":
                html_parts.append(EmailSource._decode_part(msg))

        if plain_parts:
            return "\n".join(p for p in plain_parts if p).strip()
        if html_parts:
            html = "\n".join(html_parts)
            return BeautifulSoup(html, "html.parser").get_text("\n").strip()
        return ""

    @staticmethod
    def _decode_part(part: Message) -> str:
        payload = part.get_payload(decode=True)
        if payload is None:
            return ""
        if isinstance(payload, list):
            return ""
        charset = part.get_content_charset() or "utf-8"
        try:
            return payload.decode(charset, errors="replace")
        except (LookupError, AttributeError):
            return payload.decode("utf-8", errors="replace") if isinstance(
                payload, (bytes, bytearray)
            ) else str(payload)
