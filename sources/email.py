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
from tqdm import tqdm

DEFAULT_IMAP_FOLDERS: tuple[str, ...] = (
    "INBOX",
    "[Gmail]/Sent Mail",
    "[Gmail]/Verzonden berichten",
    "Sent",
)


class EmailSource:
    """Loads email into LlamaIndex Documents from an mbox file or IMAP server."""

    collection_name: str = "emails"

    def load(
        self,
        mbox: str | None = None,
        imap: dict[str, Any] | None = None,
        skip_ids: set[str] | frozenset[str] | None = None,
        **_: Any,
    ) -> list[Document]:
        if mbox and imap:
            raise ValueError("Pass either `mbox` or `imap`, not both.")
        if mbox:
            return self.load_from_mbox(mbox, skip_ids=skip_ids)
        if imap:
            return self.load_from_imap(**imap, skip_ids=skip_ids)
        raise ValueError("EmailSource.load requires `mbox=...` or `imap={...}`.")

    # ----- mbox ---------------------------------------------------------------

    def load_from_mbox(
        self,
        path: str,
        skip_ids: set[str] | frozenset[str] | None = None,
    ) -> list[Document]:
        box = mailbox.mbox(path)
        messages = list(box)
        seen: set[str] = set()
        docs: list[Document] = []
        for msg in tqdm(messages, desc="Reading mbox", unit="msg"):
            doc = self._build_doc(msg, folder="mbox")
            if doc.doc_id in seen:
                continue
            if skip_ids and doc.doc_id in skip_ids:
                seen.add(doc.doc_id)
                continue
            seen.add(doc.doc_id)
            docs.append(doc)
        return docs

    # ----- IMAP ---------------------------------------------------------------

    def list_imap_folders(self, host: str, user: str, password: str) -> list[str]:
        """Return sorted list of all folders visible on the IMAP server."""
        with imaplib.IMAP4_SSL(host) as conn:
            conn.login(user, password)
            folders = self._list_folders(conn)
            conn.logout()
        return sorted(folders)

    _HEADER_BATCH = 500   # IDs per BODY[HEADER] round trip (lightweight)
    _RFC822_BATCH = 100   # IDs per RFC822 round trip

    def load_from_imap(
        self,
        host: str,
        user: str,
        password: str,
        folders: Iterable[str] = DEFAULT_IMAP_FOLDERS,
        limit: int | None = None,
        save_mbox: str | None = None,
        skip_ids: set[str] | frozenset[str] | None = None,
    ) -> list[Document]:
        docs: list[Document] = []
        seen: set[str] = set()
        mbox_writer = mailbox.mbox(save_mbox, create=True) if save_mbox else None
        with imaplib.IMAP4_SSL(host) as conn:
            conn.login(user, password)
            tqdm.write(f"[email] Logged in as {user}")
            available_folders = self._list_folders(conn)
            _printed_available = False
            for folder in folders:
                if folder not in available_folders:
                    if not _printed_available:
                        tqdm.write(
                            f"[email] Available folders: {', '.join(sorted(available_folders))}"
                        )
                        _printed_available = True
                    tqdm.write(f"[email] Skipping '{folder}' (not found on server)")
                    continue
                typ, _ = conn.select(f'"{folder}"', readonly=True)
                if typ != "OK":
                    tqdm.write(f"[email] Could not select '{folder}', skipping")
                    continue
                typ, data = conn.search(None, "ALL")
                if typ != "OK" or not data or not data[0]:
                    tqdm.write(f"[email] No messages in '{folder}'")
                    continue
                ids = data[0].split()
                if limit is not None:
                    ids = ids[-limit:]

                # Pre-scan Message-ID headers to skip already-indexed messages.
                if skip_ids:
                    tqdm.write(
                        f"[email] Pre-scanning {len(ids)} header(s) in '{folder}' "
                        f"to skip already-indexed messages..."
                    )
                    seq_to_mid = self._fetch_message_ids(conn, ids, self._HEADER_BATCH)
                    before = len(ids)
                    ids = [
                        num for num in ids
                        if seq_to_mid.get(num, "\x00") not in skip_ids
                    ]
                    skipped = before - len(ids)
                    if skipped:
                        tqdm.write(
                            f"[email] Skipping {skipped} already-indexed message(s) "
                            f"in '{folder}'"
                        )

                if not ids:
                    tqdm.write(f"[email] Nothing new in '{folder}'")
                    continue

                tqdm.write(f"[email] Fetching '{folder}': {len(ids)} message(s)")
                with tqdm(total=len(ids), desc=f"  {folder}", unit="msg", leave=True) as pbar:
                    for i in range(0, len(ids), self._RFC822_BATCH):
                        batch = ids[i : i + self._RFC822_BATCH]
                        seq = ",".join(n.decode() for n in batch)
                        typ, msg_data = conn.fetch(seq, "(RFC822)")
                        if typ != "OK" or not msg_data:
                            pbar.update(len(batch))
                            continue
                        for part in msg_data:
                            if not isinstance(part, tuple) or len(part) < 2:
                                continue
                            raw = part[1]
                            if not isinstance(raw, (bytes, bytearray)):
                                continue
                            if mbox_writer is not None:
                                mbox_writer.add(mailbox.mboxMessage(raw))
                            msg = stdlib_email.message_from_bytes(raw)
                            doc = self._build_doc(msg, folder=folder)
                            if doc.doc_id in seen:
                                continue
                            seen.add(doc.doc_id)
                            docs.append(doc)
                        pbar.update(len(batch))
            conn.logout()
        if mbox_writer is not None:
            mbox_writer.flush()
            mbox_writer.close()
            tqdm.write(f"[email] Messages saved to {save_mbox}")
        tqdm.write(f"[email] Fetch complete: {len(docs)} unique message(s) loaded")
        return docs

    @staticmethod
    def _fetch_message_ids(
        conn: imaplib.IMAP4_SSL,
        ids: list[bytes],
        batch_size: int,
    ) -> dict[bytes, str]:
        """Batch-fetch Message-ID headers. Returns {imap_seq_bytes: message_id_str}."""
        result: dict[bytes, str] = {}
        for i in range(0, len(ids), batch_size):
            batch = ids[i : i + batch_size]
            seq = ",".join(n.decode() for n in batch)
            typ, data = conn.fetch(seq, "(BODY[HEADER.FIELDS (MESSAGE-ID)])")
            if typ != "OK" or not data:
                continue
            for part in data:
                if not isinstance(part, tuple) or len(part) < 2:
                    continue
                info, header_bytes = part[0], part[1]
                if not isinstance(info, bytes) or not isinstance(header_bytes, (bytes, bytearray)):
                    continue
                seq_num = info.split()[0]  # e.g. b'42'
                msg = stdlib_email.message_from_bytes(header_bytes)
                mid = str(make_header(decode_header(msg.get("Message-Id", "")))).strip()
                if mid:
                    result[seq_num] = mid
        return result

    @staticmethod
    def _list_folders(conn: imaplib.IMAP4_SSL) -> set[str]:
        typ, data = conn.list()
        if typ != "OK" or not data:
            return set()
        names: set[str] = set()
        for entry in data:
            if not entry:
                continue
            decoded = entry.decode("utf-8", errors="replace").strip()
            # IMAP LIST: (attributes) "sep" mailbox-name  — name may be quoted or not
            if decoded.endswith('"'):
                # Quoted name: extract between the last pair of quotes
                last = decoded.rfind('"', 0, len(decoded) - 1)
                if last >= 0:
                    names.add(decoded[last + 1 : -1])
            else:
                # Unquoted name: last whitespace-separated token
                parts = decoded.rsplit(None, 1)
                if len(parts) == 2:
                    names.add(parts[-1])
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
        meta = {
            "source_type": "email",
            "message_id": message_id,
            "subject": subject,
            "from_addr": from_addr,
            "from_name": from_name,
            "to_addr": to_addr,
            "date": date,
            "folder": folder,
        }
        return Document(
            text=text,
            doc_id=message_id,
            metadata=meta,
            excluded_embed_metadata_keys=list(meta.keys()),
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
