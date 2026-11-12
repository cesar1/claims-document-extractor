import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session
from test_extract import sample_invoice_result

from app import main
from app.db import DocStatus, Document, make_engine
from app.extract import Extraction, ExtractionError

PDF = b"%PDF-1.4\nfake claim document\n%%EOF"


class FakeExtractor:
    """Stands in for extract_pdf so no test ever calls Claude."""

    def __init__(self):
        self.calls = 0
        self.error: Exception | None = None

    def __call__(self, pdf_bytes: bytes) -> Extraction:
        self.calls += 1
        assert pdf_bytes.startswith(PDF)
        if self.error:
            raise self.error
        return Extraction(
            result=sample_invoice_result(),
            input_tokens=10_000,
            output_tokens=3_000,
            cost_usd=0.10,
            message_id=f"msg_{self.calls}",
            model="claude-opus-5-5",
        )


@pytest.fixture
def extractor():
    return FakeExtractor()


@pytest.fixture
def client(tmp_path, extractor):
    with TestClient(main.create_app(data_dir=tmp_path, extractor=extractor)) as c:
        yield c


def upload(client, data=PDF, filename="invoice.pdf") -> dict:
    r = client.post("/documents", files={"file": (filename, data, "application/pdf")})
    assert r.status_code == 201, r.text
    return r.json()


def test_upload_runs_pipeline_and_lands_in_review(client, extractor):
    created = upload(client, filename="C:\\claims\\invoice.pdf")
    assert created["status"] == "queued"
    assert created["filename"] == "invoice.pdf"

    doc = client.get(f"/documents/{created['id']}").json()
    assert extractor.calls == 1
    assert doc["status"] == "needs_review"
    assert doc["doc_type"] == "invoice"
    assert doc["review"] is None
    assert doc["created_at"].endswith(("Z", "+00:00"))

    ext = doc["extraction"]
    assert ext["result"]["doc_type"] == "invoice"
    assert (ext["model"], ext["cost_usd"]) == ("claude-opus-5-5", 0.10)
    # the sample's one line (25.00) doesn't add up to its printed total (27.00)
    rules = {f["rule"] for f in ext["validation"]["flags"]}
    assert "lines_mismatch" in rules
    assert doc["errors"] == ext["validation"]["errors"] == 1

    pdf = client.get(doc["pdf_url"])
    assert pdf.content == PDF
    assert pdf.headers["content-type"] == "application/pdf"


def test_queue_lists_newest_first_and_filters_by_status(client, extractor):
    first = upload(client, filename="a.pdf")
    extractor.error = ExtractionError("Model declined the document")
    second = upload(client, data=PDF + b" second file", filename="b.pdf")

    assert [d["id"] for d in client.get("/documents").json()] == [second["id"], first["id"]]
    failed = client.get("/documents", params={"status": "failed"}).json()
    assert [d["id"] for d in failed] == [second["id"]]
    assert client.get("/documents", params={"status": "bogus"}).status_code == 422


def test_same_file_uploaded_twice_is_extracted_once(client, extractor):
    first = upload(client, filename="estimate.pdf")

    again = client.post("/documents", files={"file": ("copy.pdf", PDF, "application/pdf")})
    assert again.status_code == 200
    assert again.json()["id"] == first["id"]
    assert again.json()["status"] == "needs_review"
    assert extractor.calls == 1
    assert len(client.get("/documents").json()) == 1

    # different bytes are a different document
    other = upload(client, data=PDF + b" revised", filename="estimate.pdf")
    assert other["id"] != first["id"]
    assert extractor.calls == 2


def test_upload_rejects_non_pdf_and_oversized(client, extractor, monkeypatch):
    r = client.post("/documents", files={"file": ("notes.txt", b"hello", "text/plain")})
    assert r.status_code == 415

    monkeypatch.setattr(main, "MAX_UPLOAD_BYTES", 16)
    r = client.post("/documents", files={"file": ("big.pdf", PDF, "application/pdf")})
    assert r.status_code == 413

    assert extractor.calls == 0
    assert client.get("/documents").json() == []


def test_failed_extraction_is_recorded_and_can_be_retried(client, extractor):
    extractor.error = ExtractionError("Output hit max_tokens")
    doc_id = upload(client)["id"]

    doc = client.get(f"/documents/{doc_id}").json()
    assert doc["status"] == "failed"
    assert "max_tokens" in doc["error"]
    assert doc["extraction"] is None
    assert client.put(f"/documents/{doc_id}/review", json={}).status_code == 409

    extractor.error = None
    r = client.post(f"/documents/{doc_id}/reextract")
    assert r.status_code == 202
    doc = client.get(f"/documents/{doc_id}").json()
    assert (doc["status"], doc["error"]) == ("needs_review", None)
    assert extractor.calls == 2


def test_review_draft_approve_export_and_reopen(client):
    doc_id = upload(client)["id"]
    url = f"/documents/{doc_id}"
    assert client.get(f"{url}/export").status_code == 409

    # draft: fix the misread total, which clears the line-sum error
    r = client.put(f"{url}/review", json={"fields": {"total": "25.00"}})
    assert r.status_code == 200, r.text
    doc = r.json()
    assert doc["status"] == "needs_review"
    assert doc["errors"] == 0
    assert doc["review"]["approved_at"] is None
    assert [(c["name"], c["before"], c["after"]) for c in doc["review"]["changes"]] == [
        ("total", "27.00", "25.00")
    ]
    # the model's output is kept untouched next to the corrected data
    assert doc["extraction"]["validation"]["errors"] == 1
    assert client.get(f"{url}/export").status_code == 409

    # corrections are relative to the model output, so the approval repeats the fix
    corrections = {"fields": {"total": "25.00", "vendor_name": "Acme Restoration"}}
    doc = client.put(f"{url}/review", json={**corrections, "approve": True}).json()
    assert doc["status"] == "approved"
    assert doc["review"]["approved_at"] is not None
    assert len(doc["review"]["changes"]) == 2

    export = client.get(f"{url}/export").json()
    assert export["doc_type"] == "invoice"
    assert export["fields"]["total"] == "25.00"
    assert export["fields"]["vendor_name"] == "Acme Restoration"
    assert export["fields"]["claim_number"] == "CLM-1001"
    assert export["line_items"] == [
        {
            "description": "Drywall",
            "quantity": 10.0,
            "unit": "SF",
            "unit_price": 2.5,
            "amount": 25.0,
            "category": "Interior",
        }
    ]
    assert export["corrections"] == 2
    assert export["source"]["message_id"] == "msg_1"
    # still-missing required fields travel with the export
    assert {f["rule"] for f in export["open_flags"]} == {"required_missing"}

    assert client.post(f"{url}/reextract").status_code == 409

    doc = client.put(f"{url}/review", json=corrections).json()
    assert doc["status"] == "needs_review"
    assert client.get(f"{url}/export").status_code == 409


def test_approving_as_is_keeps_printed_errors_in_the_export(client):
    doc_id = upload(client)["id"]
    doc = client.put(f"/documents/{doc_id}/review", json={"approve": True}).json()
    assert doc["status"] == "approved"
    assert doc["review"]["changes"] == []

    export = client.get(f"/documents/{doc_id}/export").json()
    assert "lines_mismatch" in {f["rule"] for f in export["open_flags"]}


def test_reextract_discards_the_draft_review(client, extractor):
    doc_id = upload(client)["id"]
    client.put(f"/documents/{doc_id}/review", json={"fields": {"total": "25.00"}})

    assert client.post(f"/documents/{doc_id}/reextract").status_code == 202
    doc = client.get(f"/documents/{doc_id}").json()
    assert doc["review"] is None
    assert doc["errors"] == 1
    assert doc["extraction"]["id"] == 2
    assert extractor.calls == 2


def test_review_rejects_unknown_field_names(client):
    doc_id = upload(client)["id"]
    r = client.put(f"/documents/{doc_id}/review", json={"fields": {"not_a_field": "x"}})
    assert r.status_code == 422


@pytest.mark.parametrize(
    ("method", "path"),
    [
        ("get", ""),
        ("get", "/pdf"),
        ("get", "/export"),
        ("put", "/review"),
        ("post", "/reextract"),
    ],
)
def test_unknown_document_is_404(client, method, path):
    kwargs = {"json": {}} if method == "put" else {}
    assert client.request(method, f"/documents/nope{path}", **kwargs).status_code == 404


def test_restart_fails_interrupted_documents_without_rerunning(tmp_path, extractor):
    with TestClient(main.create_app(data_dir=tmp_path, extractor=extractor)):
        pass
    engine = make_engine(tmp_path / "extractor.db")
    with Session(engine) as session:
        session.add(Document(id="stuck", filename="a.pdf", sha256="0", size_bytes=1))
        session.add(
            Document(
                id="midway",
                filename="b.pdf",
                sha256="0",
                size_bytes=1,
                status=DocStatus.processing,
            )
        )
        session.commit()
    engine.dispose()

    with TestClient(main.create_app(data_dir=tmp_path, extractor=extractor)) as client:
        docs = client.get("/documents").json()
    assert {d["id"]: d["status"] for d in docs} == {"stuck": "failed", "midway": "failed"}
    assert all("Re-extract" in d["error"] for d in docs)
    assert extractor.calls == 0
