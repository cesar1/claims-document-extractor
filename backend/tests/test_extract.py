from types import SimpleNamespace

import pytest

from app import extract
from app.schemas import DocType, ExtractionResult, FieldName


def _field(name, value, page=1):
    return {
        "name": name,
        "value": value,
        "page": page,
        "source_text": value,
        "confidence": 0.95,
    }


def sample_invoice_result() -> ExtractionResult:
    return ExtractionResult.model_validate(
        {
            "doc_type": "invoice",
            "doc_type_confidence": 0.98,
            "fields": [
                _field("claim_number", "CLM-1001"),
                _field("date_of_loss", "2026-03-14"),
                _field("cause_of_loss", "fire"),
                _field("invoice_number", "INV-77"),
                _field("total", "27.00"),
            ],
            "line_items": [
                {
                    "description": "Drywall",
                    "quantity": 10,
                    "unit": "SF",
                    "unit_price": 2.5,
                    "amount": 25.0,
                    "category": "Interior",
                    "page": 1,
                    "confidence": 0.9,
                }
            ],
            "notes": None,
        }
    )


class FakeStream:
    def __init__(self, message):
        self._message = message

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def get_final_message(self):
        return self._message


def fake_client(message, captured: dict):
    def stream(**kwargs):
        captured.update(kwargs)
        return FakeStream(message)

    return SimpleNamespace(beta=SimpleNamespace(messages=SimpleNamespace(stream=stream)))


def fake_message(parsed, stop_reason="end_turn"):
    usage = SimpleNamespace(
        input_tokens=10_000,
        output_tokens=3_000,
        cache_read_input_tokens=0,
        cache_creation_input_tokens=0,
    )
    return SimpleNamespace(
        id="msg_test",
        stop_reason=stop_reason,
        stop_details=None,
        parsed_output=parsed,
        usage=usage,
    )


def test_field_lookup():
    r = sample_invoice_result()
    assert r.doc_type is DocType.invoice
    assert r.value(FieldName.invoice_number) == "INV-77"
    assert r.field(FieldName.vendor_name) is None


def test_schema_is_accepted_by_sdk_structured_output_transform():
    from anthropic.lib._parse._transform import to_json_output_format

    schema = to_json_output_format(ExtractionResult)["schema"]
    assert schema["type"] == "object"
    assert "doc_type" in schema["required"]


def test_extract_pdf_success_and_cost():
    captured = {}
    client = fake_client(fake_message(sample_invoice_result()), captured)
    ex = extract.extract_pdf(b"%PDF-1.4 fake", client=client)

    assert ex.result.doc_type is DocType.invoice
    assert ex.cost_usd == pytest.approx((10_000 * 4 + 3_000 * 20) / 1e6)
    assert captured["fallbacks"] == "default"
    assert captured["output_format"] is ExtractionResult
    doc_block = captured["messages"][0]["content"][0]
    assert doc_block["type"] == "document"
    assert doc_block["source"]["media_type"] == "application/pdf"


@pytest.mark.parametrize("stop_reason", ["refusal", "max_tokens"])
def test_extract_pdf_raises_on_bad_stop(stop_reason):
    client = fake_client(fake_message(None, stop_reason=stop_reason), {})
    with pytest.raises(extract.ExtractionError):
        extract.extract_pdf(b"%PDF", client=client)
