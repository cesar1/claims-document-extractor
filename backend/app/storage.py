"""Uploaded PDFs on local disk, named by document id (never by the client's filename)."""

from pathlib import Path


class Storage:
    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)

    def path(self, doc_id: str) -> Path:
        return self.root / f"{doc_id}.pdf"

    def save(self, doc_id: str, data: bytes) -> None:
        self.path(doc_id).write_bytes(data)

    def read(self, doc_id: str) -> bytes:
        return self.path(doc_id).read_bytes()
