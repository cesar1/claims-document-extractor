"""Run the extractor over a labeled sample set and score it against ground truth.

Usage (from repo root, with the backend venv):
    backend/.venv/Scripts/python scripts/eval.py run samples/synthetic            # cost estimate only
    backend/.venv/Scripts/python scripts/eval.py run samples/synthetic --yes      # spend: Batches API (50% off)
    backend/.venv/Scripts/python scripts/eval.py run samples/synthetic --mode live --limit 3 --yes
    backend/.venv/Scripts/python scripts/eval.py resume data/eval/<run>           # collect a pending batch
    backend/.venv/Scripts/python scripts/eval.py score data/eval/<run>            # re-score saved predictions, free

Each run writes data/eval/<timestamp>-<mode>/ with predictions/<id>.json, report.json and report.md.
"""

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

import anthropic
from app import extract
from app.schemas import ExtractionResult
from app.scoring import score_document, summarize
from app.validation import validate
from pypdf import PdfReader

# Rough token model from the real-document runs: ~2.5K input tokens/page, ~1.5K output base +
# ~120 output tokens per line item (includes billed reasoning at effort=medium).
IN_PER_PAGE, OUT_BASE, OUT_PER_LINE = 2500, 1500, 120


def load_samples(sample_dir: Path, pattern: str, limit: int | None) -> list[tuple[str, Path, dict]]:
    out = []
    for truth_path in sorted(sample_dir.glob(f"{pattern}.truth.json")):
        doc_id = truth_path.name.removesuffix(".truth.json")
        pdf = truth_path.with_name(f"{doc_id}.pdf")
        if pdf.exists():
            out.append((doc_id, pdf, json.loads(truth_path.read_text())))
    return out[:limit] if limit else out


def estimate_cost(samples, batch: bool) -> float:
    total = 0.0
    for _, pdf, truth in samples:
        pages = len(PdfReader(pdf).pages)
        tokens_in = pages * IN_PER_PAGE
        tokens_out = OUT_BASE + OUT_PER_LINE * len(truth["line_items"])
        total += (tokens_in * extract.PRICE_INPUT + tokens_out * extract.PRICE_OUTPUT) / 1e6
    return total * (0.5 if batch else 1.0)


def save_prediction(
    run_dir: Path,
    doc_id: str,
    result: ExtractionResult | None,
    usage,
    cost: float,
    error: str | None = None,
) -> None:
    pred_dir = run_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "doc_id": doc_id,
        "error": error,
        "cost_usd": round(cost, 6),
        "input_tokens": getattr(usage, "input_tokens", None),
        "output_tokens": getattr(usage, "output_tokens", None),
        "result": result.model_dump(mode="json") if result else None,
    }
    (pred_dir / f"{doc_id}.json").write_text(json.dumps(record, indent=2))


def run_live(samples, run_dir: Path, client) -> None:
    for doc_id, pdf, _ in samples:
        try:
            ex = extract.extract_pdf(pdf.read_bytes(), client=client)
            save_prediction(run_dir, doc_id, ex.result, ex, ex.cost_usd)
            print(f"[ok] {doc_id} ${ex.cost_usd:.4f}")
        except (extract.ExtractionError, anthropic.APIError) as e:
            save_prediction(run_dir, doc_id, None, None, 0.0, error=str(e))
            print(f"[FAIL] {doc_id}: {e}")


def submit_batch(samples, run_dir: Path, client) -> str:
    batch = client.messages.batches.create(
        requests=[
            {
                "custom_id": doc_id,
                "params": extract.build_batch_params(pdf.read_bytes()),
            }
            for doc_id, pdf, _ in samples
        ]
    )
    (run_dir / "batch.json").write_text(json.dumps({"batch_id": batch.id}))
    print(f"Submitted batch {batch.id} ({len(samples)} requests)")
    return batch.id


def collect_batch(batch_id: str, run_dir: Path, client, poll_seconds: int = 30) -> None:
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        c = batch.request_counts
        print(
            f"  {batch.processing_status}: processing={c.processing} succeeded={c.succeeded} errored={c.errored}"
        )
        if batch.processing_status == "ended":
            break
        time.sleep(poll_seconds)
    for item in client.messages.batches.results(batch_id):
        doc_id = item.custom_id
        if item.result.type != "succeeded":
            save_prediction(
                run_dir,
                doc_id,
                None,
                None,
                0.0,
                error=f"batch result: {item.result.type}",
            )
            print(f"[FAIL] {doc_id}: {item.result.type}")
            continue
        msg = item.result.message
        cost = extract.cost_usd(msg.usage, batch=True)
        try:
            save_prediction(run_dir, doc_id, extract.parse_message_text(msg), msg.usage, cost)
        except (extract.ExtractionError, ValueError) as e:
            save_prediction(run_dir, doc_id, None, msg.usage, cost, error=str(e))
            print(f"[FAIL] {doc_id}: {e}")


def score_run(run_dir: Path) -> dict:
    meta = json.loads((run_dir / "run.json").read_text())
    sample_dir = Path(meta["samples"])
    scores, failures, cost = [], [], 0.0
    validations = []  # (doc_id, traps, ValidationReport)
    for pred_path in sorted((run_dir / "predictions").glob("*.json")):
        rec = json.loads(pred_path.read_text())
        cost += rec["cost_usd"] or 0
        if rec["result"] is None:
            failures.append({"doc_id": rec["doc_id"], "error": rec["error"]})
            continue
        truth = json.loads((sample_dir / f"{rec['doc_id']}.truth.json").read_text())
        result = ExtractionResult.model_validate(rec["result"])
        scores.append(score_document(rec["doc_id"], truth, result))
        validations.append((rec["doc_id"], truth.get("traps", []), validate(result)))

    report = {
        "run": run_dir.name,
        **meta,
        "cost_usd": round(cost, 4),
        "failures": failures,
        "summary": summarize(scores),
        "errors": [
            {
                "doc_id": s.doc_id,
                "field": f.name,
                "outcome": f.outcome,
                "truth": f.truth,
                "pred": f.pred,
                "confidence": f.confidence,
            }
            for s in scores
            for f in s.fields
            if not f.ok
        ],
        "validation": validation_summary(validations),
        "line_issues": [
            {
                "doc_id": s.doc_id,
                "truth": s.lines_truth,
                "pred": s.lines_pred,
                "matched": s.lines_matched,
                "detail_errors": s.line_detail_errors,
                "sum_ok": s.lines_sum_ok,
            }
            for s in scores
            if s.lines_truth
            and (
                s.lines_matched != s.lines_truth
                or s.lines_pred != s.lines_truth
                or s.line_detail_errors
                or not s.lines_sum_ok
            )
        ],
    }
    (run_dir / "report.json").write_text(json.dumps(report, indent=2, default=str))
    (run_dir / "report.md").write_text(render_markdown(report))
    return report


def validation_summary(validations) -> dict:
    """How validation behaves on the sample set: printed math errors (math_error trap) should get an
    error flag; nothing else should."""
    math_docs = [(d, rep) for d, traps, rep in validations if "math_error" in traps]
    other = [(d, rep) for d, traps, rep in validations if "math_error" not in traps]
    rule_counts: dict[str, int] = {}
    for _, _, rep in validations:
        for f in rep.flags:
            key = f"{f.severity.value}:{f.rule}"
            rule_counts[key] = rule_counts.get(key, 0) + 1
    return {
        "docs_needing_review": sum(rep.needs_review for _, _, rep in validations),
        "docs_with_errors": sum(bool(rep.errors) for _, _, rep in validations),
        "math_error_docs": len(math_docs),
        "math_error_docs_caught": sum(bool(rep.errors) for _, rep in math_docs),
        "false_error_docs": [d for d, rep in other if rep.errors],
        "rule_counts": dict(sorted(rule_counts.items())),
    }


def pct(x) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"


def trap_section(per_trap: dict) -> list[str]:
    if not per_trap:
        return []
    rows = [
        f"| {name} | {t['docs']} | {pct(t['target_field_accuracy'])} ({t['target_fields']}) | "
        f"{pct(t['line_recall']) if t['line_recall'] is not None else '-'} |"
        for name, t in per_trap.items()
    ]
    return [
        "## Per trap",
        "",
        "| Trap | Docs | Target field accuracy | Line recall |",
        "|---|---|---|---|",
        *rows,
        "",
    ]


def render_markdown(r: dict) -> str:
    s, li, cal = r["summary"], r["summary"]["line_items"], r["summary"]["calibration"]
    lines = [
        f"# Eval {r['run']}",
        "",
        (
            f"Model `{r['model']}`, effort `{r['effort']}`, mode `{r['mode']}`. "
            f"{s['documents']} scored, {len(r['failures'])} failed. Cost ${r['cost_usd']:.2f}."
        ),
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Doc type accuracy | {pct(s['doc_type_accuracy'])} |",
        f"| Field accuracy (all) | {pct(s['field_accuracy'])} |",
        f"| Field accuracy (clean / scan) | {pct(s['field_accuracy_clean'])} / {pct(s['field_accuracy_scan'])} |",
        f"| Line item recall / precision | {pct(li['recall'])} / {pct(li['precision'])} |",
        f"| Line item detail accuracy | {pct(li['detail_accuracy'])} |",
        f"| Docs whose lines reconcile | {li['docs_sum_reconciles']} / {li['docs_with_lines']} |",
        f"| Hallucinated fields | {s['outcomes']['hallucinated']} |",
        (
            f"| Accuracy at confidence >= 0.9 / < 0.9 | {pct(cal['high_conf_accuracy'])} ({cal['high_conf_fields']}) / "
            f"{pct(cal['low_conf_accuracy'])} ({cal['low_conf_fields']}) |"
        ),
        "",
        *trap_section(s.get("per_trap", {})),
        "## Per doc type",
        "",
        *[f"- {t}: {pct(a)}" for t, a in s["per_doc_type"].items()],
        "",
        "## Per field",
        "",
        *[f"- {n}: {pct(a)}" for n, a in s["per_field"].items()],
        "",
        f"## Field errors ({len(r['errors'])})",
        "",
        *[
            f"- `{e['doc_id']}` {e['field']} ({e['outcome']}, conf {e['confidence']}): "
            f"truth={e['truth']!r} pred={e['pred']!r}"
            for e in r["errors"]
        ],
        "",
        f"## Line item issues ({len(r['line_issues'])})",
        "",
        *[
            f"- `{x['doc_id']}` truth={x['truth']} pred={x['pred']} matched={x['matched']} "
            f"detail_errors={x['detail_errors']} sum_ok={x['sum_ok']}"
            for x in r["line_issues"]
        ],
    ]
    v = r.get("validation")
    if v:
        lines += [
            "",
            "## Validation",
            "",
            f"- Documents needing review: {v['docs_needing_review']} / {s['documents']}",
            f"- Printed math errors caught: {v['math_error_docs_caught']} / {v['math_error_docs']}",
            (
                f"- Error flags on documents without a planted error: {len(v['false_error_docs'])} "
                f"{v['false_error_docs'] or ''}"
            ),
            "",
            *[f"- `{rule}`: {n}" for rule, n in v["rule_counts"].items()],
        ]
    if r["failures"]:
        lines += [
            "",
            "## Failures",
            "",
            *[f"- `{f['doc_id']}`: {f['error']}" for f in r["failures"]],
        ]
    return "\n".join(lines) + "\n"


def print_summary(report: dict, run_dir: Path) -> None:
    s = report["summary"]
    print(
        f"\nScored {s['documents']} docs ({len(report['failures'])} failed), cost ${report['cost_usd']:.2f}"
    )
    print(
        f"  field accuracy {pct(s['field_accuracy'])} (clean {pct(s['field_accuracy_clean'])}, "
        f"scan {pct(s['field_accuracy_scan'])})"
    )
    print(f"  line recall {pct(s['line_items']['recall'])}, precision {pct(s['line_items']['precision'])}")
    print(f"  report: {run_dir / 'report.md'}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("samples", type=Path)
    r.add_argument("--mode", choices=["batch", "live"], default="batch")
    r.add_argument("--filter", default="*", help="glob on doc id, e.g. 'invoice-*'")
    r.add_argument("--limit", type=int)
    r.add_argument(
        "--yes",
        action="store_true",
        help="actually call the API (otherwise estimate only)",
    )
    sub.add_parser("resume").add_argument("run_dir", type=Path)
    sub.add_parser("score").add_argument("run_dir", type=Path)
    args = ap.parse_args()

    if args.cmd == "score":
        print_summary(score_run(args.run_dir), args.run_dir)
        return 0

    client = anthropic.Anthropic()
    if args.cmd == "resume":
        batch_id = json.loads((args.run_dir / "batch.json").read_text())["batch_id"]
        collect_batch(batch_id, args.run_dir, client)
        print_summary(score_run(args.run_dir), args.run_dir)
        return 0

    samples = load_samples(args.samples, args.filter, args.limit)
    if not samples:
        print(f"No samples matching {args.filter!r} in {args.samples}")
        return 1
    est = estimate_cost(samples, batch=args.mode == "batch")
    print(f"{len(samples)} documents, mode={args.mode}, model={extract.MODEL}, effort={extract.EFFORT}")
    print(f"Estimated cost: ${est:.2f}")
    if not args.yes:
        print("Dry run. Re-run with --yes to call the API.")
        return 0

    run_dir = ROOT / "data" / "eval" / f"{datetime.now():%Y%m%d-%H%M%S}-{args.mode}"
    run_dir.mkdir(parents=True)
    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "samples": str(args.samples.resolve()),
                "mode": args.mode,
                "model": extract.MODEL,
                "effort": extract.EFFORT,
                "documents": [d for d, _, _ in samples],
                "estimated_cost_usd": round(est, 4),
            },
            indent=2,
        )
    )
    if args.mode == "live":
        run_live(samples, run_dir, client)
    else:
        collect_batch(submit_batch(samples, run_dir, client), run_dir, client)
    print_summary(score_run(run_dir), run_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
