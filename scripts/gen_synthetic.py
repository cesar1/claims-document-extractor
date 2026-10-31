"""Generate synthetic claim PDFs with ground-truth JSON for the extraction eval.

Usage (from repo root, with the backend venv):
    backend/.venv/Scripts/python scripts/gen_synthetic.py                    # easy set -> samples/synthetic
    backend/.venv/Scripts/python scripts/gen_synthetic.py --profile hard     # hard set -> samples/synthetic-hard

Writes <id>.pdf and <id>.truth.json per document. Some documents are "scan-like": rasterized,
rotated, noisy, JPEG-compressed image PDFs with no text layer. Output is deterministic for a
given profile, seed and library versions.

The hard profile adds traps modeled on real Xactimate estimates and messy paperwork; each truth
file lists the traps its document contains so the eval can report accuracy per trap:
    xactimate          line totals include per-line tax and O&P; room total rows; split summary
    long_table         40-70 line items spanning several pages
    math_error         printed total doesn't equal the arithmetic (truth = what's printed)
    conflicting_dates  extra dates near the real one (inspected, received, printed, due, report)
    stamp              RECEIVED / PAID stamp with its own date
    label_variants     DOL, Clm #, Policyholder, Balance Due, ...
    claim_in_text      claim/policy numbers in a "Re:" sentence instead of labeled rows
    credit_line        negative credit/discount line on an invoice
    time_distractors   dispatch and clear times next to alarm/arrival times
    time_format        times printed as "1435 hrs"
"""

import argparse
import io
import json
import random
import sys
from datetime import date, datetime, timedelta
from html import escape
from pathlib import Path

from faker import Faker
from PIL import Image, ImageFilter
from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "backend"))
from app.schemas import EXPECTED_FIELDS, DocType

STYLES = getSampleStyleSheet()
SMALL = STYLES["BodyText"].clone("small", fontSize=8.5, leading=10.5)

ROOMS = [
    "Kitchen",
    "Living Room",
    "Master Bedroom",
    "Bedroom 2",
    "Hallway",
    "Bathroom",
    "Garage",
]
ESTIMATE_ITEMS = [
    ('Remove drywall - 1/2"', "SF", 0.45, 1.20),
    ('Install drywall - 1/2" hung, taped, floated', "SF", 1.85, 3.10),
    ("Seal/prime then paint walls (2 coats)", "SF", 0.85, 1.40),
    ("Remove carpet pad", "SF", 0.20, 0.45),
    ("Carpet - mid grade", "SF", 3.20, 5.10),
    ('Baseboard - 3 1/4" MDF', "LF", 2.40, 4.10),
    ("Water extraction from carpeted floor", "SF", 0.55, 0.95),
    ("Dehumidifier (per 24 hour period)", "DA", 65.0, 95.0),
    ("Air mover (per 24 hour period)", "DA", 25.0, 38.0),
    ("Apply antimicrobial agent", "SF", 0.22, 0.40),
    ("Smoke odor treatment - thermal fog", "SF", 0.30, 0.55),
    ("Clean ceiling - heavy smoke", "SF", 0.60, 0.95),
    ("Interior door - slab only", "EA", 145.0, 260.0),
    ("R&R kitchen base cabinet", "LF", 185.0, 290.0),
    ("Roofing - laminated comp. shingle", "SQ", 245.0, 330.0),
    ("Haul debris - per pickup truck load", "EA", 140.0, 210.0),
    ("Labor - general laborer", "HR", 48.0, 72.0),
]
INVOICE_ITEMS = [
    ("Emergency board-up service", "EA", 350.0, 650.0),
    ("Water mitigation - day rate", "DA", 400.0, 750.0),
    ("Contents pack-out", "HR", 55.0, 85.0),
    ("Storage container rental", "MO", 175.0, 260.0),
    ("Plumbing repair - supply line", "EA", 180.0, 420.0),
    ("Electrical inspection", "EA", 150.0, 300.0),
    ("Dumpster - 20 yd", "EA", 420.0, 610.0),
    ("Replacement window - vinyl double hung", "EA", 310.0, 520.0),
    ("Labor - technician", "HR", 60.0, 95.0),
]
CAUSES = {
    "water": [
        "Water Damage - burst supply line",
        "Water - dishwasher leak",
        "Water heater failure",
    ],
    "fire": ["Fire - kitchen grease fire", "Fire - electrical", "Structure fire"],
    "smoke": ["Smoke damage from neighboring unit fire"],
    "wind": ["Windstorm - roof damage", "Wind - fallen tree"],
    "hail": ["Hail damage to roof and gutters"],
}
FIRE_CAUSES = [
    ("Kitchen", "Accidental - unattended cooking on range top"),
    ("Garage", "Accidental - electrical failure in extension cord"),
    ("Living Room", "Accidental - candle too close to combustibles"),
    ("Attic", "Undetermined - fire damage too extensive to determine cause"),
    ("Bedroom", "Accidental - space heater too close to bedding"),
]
LABELS = {
    "insured": ["Insured", "Insured/Owner", "Policyholder", "Customer"],
    "claim": ["Claim Number", "Claim #", "Clm No.", "Carrier Claim #"],
    "policy": ["Policy Number", "Policy #", "Pol. No."],
    "dol": ["Date of Loss", "DOL", "Loss Date"],
    "type": ["Type of Loss", "Cause of Loss", "Peril"],
    "amount_due": ["AMOUNT DUE", "Balance Due", "Please Pay"],
    "owner": ["Owner / Occupant", "Occupant", "Property Owner"],
}


# Realistic quantity range per unit of measure
QTY_RANGE = {
    "SF": (20, 650),
    "LF": (6, 60),
    "SY": (10, 90),
    "SQ": (4, 32),
    "HR": (2, 40),
    "DA": (1, 6),
    "EA": (1, 4),
    "MO": (1, 3),
}


def money(x: float) -> str:
    return f"${x:,.2f}" if x >= 0 else f"(${-x:,.2f})"


def mdY(d: date) -> str:
    return d.strftime("%m/%d/%Y")


class Gen:
    def __init__(self, seed: int, hard: bool = False):
        self.r = random.Random(seed)
        self.fake = Faker("en_US")
        self.fake.seed_instance(seed)
        self.hard = hard
        self.traps: list[str] = []

    def trap(self, name: str, p: float) -> bool:
        """Hard profile only: decide whether this document gets a trap, and record it.
        Never draws from the RNG in the easy profile, so the easy set stays reproducible."""
        if not self.hard or self.r.random() >= p:
            return False
        self.traps.append(name)
        return True

    def label(self, key: str, default: str) -> str:
        return self.r.choice(LABELS[key]) if "label_variants" in self.traps else default

    def name(self) -> str:
        return f"{self.fake.first_name()} {self.fake.last_name()}"

    def address(self) -> str:
        # Faker's address() sometimes yields military APO/DPO formats; build a civilian one
        f = self.fake
        return f"{f.street_address()}, {f.city()}, {f.state_abbr(include_territories=False)} {f.postcode()}"

    def maybe(self, value, p_present: float = 0.8):
        return value if self.r.random() < p_present else None

    def common(self) -> dict:
        loss = date(2026, 1, 1) + timedelta(days=self.r.randint(0, 240))
        cause = self.r.choice(list(CAUSES))
        return {
            "claim_number": self.maybe(f"CLM-{self.r.randint(10**6, 10**7 - 1)}", 0.75),
            "policy_number": self.maybe(
                f"HO{self.r.randint(100, 999)}-{self.r.randint(10**5, 10**6 - 1)}", 0.6
            ),
            "insured_name": self.name(),
            "property_address": self.address(),
            "date_of_loss": self.maybe(loss, 0.85),
            "cause_of_loss": cause,
            "cause_of_loss_detail": self.r.choice(CAUSES[cause]),
            "_loss": loss,
        }

    def lines(self, catalog, n: int, rooms: list[str] | None) -> list[dict]:
        out = []
        for _ in range(n):
            desc, unit, lo, hi = self.r.choice(catalog)
            lo_q, hi_q = QTY_RANGE[unit]
            if unit in ("EA", "DA", "MO"):
                qty = self.r.randint(lo_q, hi_q)
            else:
                qty = round(self.r.uniform(lo_q, hi_q), 2 if unit != "HR" else 1)
            price = round(self.r.uniform(lo, hi), 2)
            out.append(
                {
                    "description": desc,
                    "quantity": qty,
                    "unit": unit,
                    "unit_price": price,
                    "amount": round(qty * price, 2),
                    "category": self.r.choice(rooms) if rooms else None,
                }
            )
        if rooms:
            out.sort(key=lambda li: rooms.index(li["category"]))
        return out

    def math_error(self) -> float:
        """A plausible arithmetic slip to add to a printed total."""
        return round(self.r.choice([-1, 1]) * self.r.uniform(5, 250), 2)


HEADER_BG = colors.HexColor("#dde3ea")


def table_style(header_bg=HEADER_BG):
    return TableStyle(
        [
            ("FONT", (0, 0), (-1, -1), "Helvetica", 8.5),
            ("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8.5),
            ("BACKGROUND", (0, 0), (-1, 0), header_bg),
            ("LINEBELOW", (0, 0), (-1, 0), 0.5, colors.black),
            ("ALIGN", (1, 1), (-1, -1), "RIGHT"),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
        ]
    )


def kv_table(rows: list[tuple[str, str]], col_widths=(1.6 * inch, 4.9 * inch)) -> Table:
    t = Table(
        [[Paragraph(f"<b>{k}</b>", SMALL), Paragraph(v, SMALL)] for k, v in rows],
        colWidths=col_widths,
    )
    t.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ]
        )
    )
    return t


def stamp_painter(text: str, stamp_date: date, r: random.Random):
    """onFirstPage callback drawing a rotated rubber-stamp box in red."""
    angle = r.uniform(-14, 14)
    x, y = r.uniform(4.2, 5.6) * inch, r.uniform(8.4, 9.6) * inch

    def paint(canvas, _doc):
        canvas.saveState()
        canvas.translate(x, y)
        canvas.rotate(angle)
        canvas.setStrokeColor(colors.HexColor("#b3261e"))
        canvas.setFillColor(colors.HexColor("#b3261e"))
        canvas.setLineWidth(2)
        canvas.roundRect(-0.9 * inch, -0.38 * inch, 1.8 * inch, 0.76 * inch, 6)
        canvas.setFont("Helvetica-Bold", 16)
        canvas.drawCentredString(0, 0.05 * inch, text)
        canvas.setFont("Helvetica-Bold", 11)
        canvas.drawCentredString(0, -0.25 * inch, stamp_date.strftime("%b %d %Y").upper())
        canvas.restoreState()

    return paint


def build_pdf(story, on_first_page=None) -> bytes:
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=LETTER,
        leftMargin=0.7 * inch,
        rightMargin=0.7 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
    )
    if on_first_page:
        doc.build(story, onFirstPage=on_first_page)
    else:
        doc.build(story)
    return buf.getvalue()


def header_rows(c: dict, g: Gen) -> list[tuple[str, str]]:
    rows = [
        (g.label("insured", "Insured"), c["insured_name"]),
        ("Property", c["property_address"]),
    ]
    if "claim_in_text" not in g.traps:
        if c["claim_number"]:
            rows.append((g.label("claim", "Claim Number"), c["claim_number"]))
        if c["policy_number"]:
            rows.append((g.label("policy", "Policy Number"), c["policy_number"]))
    if c["date_of_loss"]:
        rows.append((g.label("dol", "Date of Loss"), mdY(c["date_of_loss"])))
    rows.append((g.label("type", "Type of Loss"), c["cause_of_loss_detail"]))
    return rows


def claim_reference(c: dict, g: Gen) -> list:
    """claim_in_text trap: put the numbers in a sentence instead of labeled rows."""
    if "claim_in_text" not in g.traps:
        return []
    if not c["claim_number"]:
        c["claim_number"] = f"CLM-{g.r.randint(10**6, 10**7 - 1)}"
    ref = f"Re: your claim {c['claim_number']}"
    if c["policy_number"]:
        ref += f" under policy {c['policy_number']}"
    return [Paragraph(ref + ". Please reference this number on all correspondence.", SMALL), Spacer(1, 6)]


def estimate_room_tables(g: Gen, items: list[dict], rooms: list[str], numbered: bool) -> list:
    story = []
    for room in rooms:
        story.append(Paragraph(f"<b>{room}</b>", STYLES["Heading4"]))
        rows = [
            ["#", "Description", "Qty", "Unit", "Unit Price", "Total"]
            if numbered
            else ["Description", "Quantity", "Unit Price", "Line Total"]
        ]
        for i, li in enumerate([li for li in items if li["category"] == room], start=1):
            qty = f"{li['quantity']:g}"
            if numbered:
                rows.append(
                    [
                        str(i),
                        Paragraph(escape(li["description"]), SMALL),
                        qty,
                        li["unit"],
                        money(li["unit_price"]),
                        money(li["amount"]),
                    ]
                )
            else:
                rows.append(
                    [
                        Paragraph(escape(li["description"]), SMALL),
                        f"{qty} {li['unit']}",
                        money(li["unit_price"]),
                        money(li["amount"]),
                    ]
                )
        widths = [0.3, 3.3, 0.7, 0.5, 0.9, 1.0] if numbered else [3.7, 1.0, 0.9, 1.1]
        t = Table(rows, colWidths=[w * inch for w in widths], repeatRows=1)
        t.setStyle(table_style())
        story.append(t)
    return story


def xactimate_room_tables(items: list[dict], rooms: list[str]) -> list:
    """Xactimate-style: per-line Tax and O&P columns, TOTAL includes both, plus a room total
    row that must not be read as a line item."""
    story, n = [], 0
    for room in rooms:
        story.append(Paragraph(f"<b>{room}</b>", STYLES["Heading4"]))
        rows = [["", "DESCRIPTION", "QTY", "REPLACE", "TAX", "O&P", "TOTAL"]]
        room_items = [li for li in items if li["category"] == room]
        for li in room_items:
            n += 1
            rows.append(
                [
                    f"{n}.",
                    Paragraph(escape(li["description"]), SMALL),
                    f"{li['quantity']:g} {li['unit']}",
                    f"{li['unit_price']:,.2f}",
                    f"{li['_tax']:,.2f}",
                    f"{li['_op']:,.2f}",
                    f"{li['amount']:,.2f}",
                ]
            )
        rows.append(
            [
                "",
                Paragraph(f"<b>Totals: {room}</b>", SMALL),
                "",
                "",
                f"{sum(li['_tax'] for li in room_items):,.2f}",
                f"{sum(li['_op'] for li in room_items):,.2f}",
                f"{sum(li['amount'] for li in room_items):,.2f}",
            ]
        )
        t = Table(rows, colWidths=[w * inch for w in (0.35, 2.75, 0.85, 0.75, 0.6, 0.7, 0.9)], repeatRows=1)
        style = table_style()
        style.add("LINEABOVE", (0, -1), (-1, -1), 0.5, colors.black)
        style.add("FONT", (0, -1), (-1, -1), "Helvetica-Bold", 8.5)
        t.setStyle(style)
        story.append(t)
    return story


def totals_table(rows: list[list[str]]) -> Table:
    t = Table(rows, colWidths=[2.2 * inch, 1.3 * inch], hAlign="RIGHT")
    t.setStyle(
        TableStyle(
            [
                ("ALIGN", (1, 0), (1, -1), "RIGHT"),
                ("FONT", (0, -1), (-1, -1), "Helvetica-Bold"),
                ("LINEABOVE", (0, -1), (-1, -1), 0.8, colors.black),
            ]
        )
    )
    return t


def contractor_estimate(g: Gen) -> tuple[bytes, dict]:
    c = g.common()
    company = g.fake.company() + " Restoration"
    license_no = g.maybe(f"CRC{g.r.randint(10**5, 10**6 - 1)}", 0.7)
    est_no = f"EST-{g.r.randint(1000, 9999)}"
    est_date = c["_loss"] + timedelta(days=g.r.randint(2, 20))
    rooms = g.r.sample(ROOMS, g.r.randint(2, 4))
    long_table = g.trap("long_table", 0.4)
    items = g.lines(ESTIMATE_ITEMS, g.r.randint(40, 70) if long_table else g.r.randint(6, 28), rooms)
    xact = g.trap("xactimate", 0.5)
    g.trap("math_error", 0.3)
    g.trap("conflicting_dates", 0.6)
    stamp = g.trap("stamp", 0.4)
    g.trap("label_variants", 0.6)
    g.trap("claim_in_text", 0.3)

    tax_rate = g.r.choice([0.0625, 0.0725, 0.0825])
    if xact:
        for li in items:
            base = li["amount"]
            li["_tax"] = round(base * tax_rate * g.r.uniform(0.3, 0.6), 2)
            li["_op"] = round(base * 0.20, 2)
            li["amount"] = round(base + li["_tax"] + li["_op"], 2)
            li["_base"] = base
        line_total = round(sum(li["_base"] for li in items), 2)
        tax = round(sum(li["_tax"] for li in items), 2)
        subtotal = round(line_total + tax, 2)
        op = round(sum(li["_op"] for li in items), 2)
        total = round(subtotal + op, 2)
    else:
        subtotal = round(sum(li["amount"] for li in items), 2)
        tax = round(subtotal * tax_rate, 2)
        op = round(subtotal * 0.20, 2) if g.r.random() < 0.7 else None
        total = round(subtotal + tax + (op or 0), 2)
    if "math_error" in g.traps:
        total = round(total + g.math_error(), 2)

    story = [
        Paragraph(f"<b>{company}</b>", STYLES["Title"]),
        Paragraph(
            g.address() + (f" &nbsp;|&nbsp; License # {license_no}" if license_no else ""),
            SMALL,
        ),
        Spacer(1, 10),
    ]
    if "conflicting_dates" in g.traps:
        story += [
            Paragraph(f"<b>ESTIMATE {est_no}</b>", STYLES["Heading3"]),
            kv_table(
                [
                    ("Date Received", mdY(c["_loss"] + timedelta(days=g.r.randint(0, 1)))),
                    ("Date Inspected", mdY(c["_loss"] + timedelta(days=g.r.randint(1, 3)))),
                    ("Estimate Date", mdY(est_date)),
                ]
            ),
            Spacer(1, 4),
        ]
    else:
        story.append(
            Paragraph(
                f"<b>ESTIMATE {est_no}</b> &nbsp;&nbsp; Date: {mdY(est_date)}",
                STYLES["Heading3"],
            )
        )
    story += [*claim_reference(c, g), kv_table(header_rows(c, g)), Spacer(1, 10)]
    numbered = g.r.random() < 0.5  # two layouts so the model can't key on one template
    story += xactimate_room_tables(items, rooms) if xact else estimate_room_tables(g, items, rooms, numbered)

    if xact:
        totals = [
            ["Line Item Total", money(line_total)],
            ["Material Sales Tax", money(tax)],
            ["Subtotal", money(subtotal)],
            ["Overhead & Profit", money(op)],
            ["Replacement Cost Value", money(total)],
        ]
    else:
        totals = [["Subtotal", money(subtotal)], ["Sales Tax", money(tax)]]
        if op is not None:
            totals.append(["Overhead & Profit (20%)", money(op)])
        totals.append(["TOTAL ESTIMATE", money(total)])
    story += [Spacer(1, 12), totals_table(totals)]
    if "conflicting_dates" in g.traps:
        printed = est_date + timedelta(days=g.r.randint(10, 60))
        story += [Spacer(1, 16), Paragraph(f"Printed {mdY(printed)} &nbsp;&nbsp; {est_no}", SMALL)]

    stamp_cb = stamp_painter("RECEIVED", est_date + timedelta(days=g.r.randint(1, 9)), g.r) if stamp else None
    fields = {k: c[k] for k in c if not k.startswith("_")}
    fields.update(
        contractor_name=company,
        contractor_license=license_no,
        estimate_number=est_no,
        estimate_date=est_date,
        subtotal=subtotal,
        tax=tax,
        overhead_and_profit=op,
        total=total,
    )
    return build_pdf(story, stamp_cb), truth(DocType.contractor_estimate, fields, items, g.traps)


def invoice(g: Gen) -> tuple[bytes, dict]:
    c = g.common()
    vendor = g.fake.company()
    inv_no = f"INV-{g.r.randint(10000, 99999)}"
    inv_date = c["_loss"] + timedelta(days=g.r.randint(1, 30))
    items = g.lines(INVOICE_ITEMS, g.r.randint(2, 9), None)
    if g.trap("credit_line", 0.4):
        credit = round(g.r.uniform(50, 400), 2)
        desc = g.r.choice(["Courtesy discount", "Credit - duplicate charge reversed", "Insurance adjustment"])
        items.append(
            {
                "description": desc,
                "quantity": 1,
                "unit": "EA",
                "unit_price": -credit,
                "amount": -credit,
                "category": None,
            }
        )
    g.trap("math_error", 0.3)
    g.trap("conflicting_dates", 0.6)
    stamp = g.trap("stamp", 0.5)
    g.trap("label_variants", 0.6)
    g.trap("claim_in_text", 0.3)

    subtotal = round(sum(li["amount"] for li in items), 2)
    tax = round(subtotal * g.r.choice([0.0, 0.0625, 0.08]), 2)
    total = round(subtotal + tax, 2)
    if "math_error" in g.traps:
        total = round(total + g.math_error(), 2)
    deposit = round(total * 0.3, 2) if g.r.random() < 0.35 else 0.0
    amount_due = round(total - deposit, 2)

    story = [
        Paragraph(f"<b>{vendor}</b>", STYLES["Title"]),
        Paragraph(g.address() + " &nbsp;|&nbsp; " + g.fake.phone_number(), SMALL),
        Spacer(1, 10),
        Paragraph(
            f"<b>INVOICE</b> &nbsp; No. {inv_no} &nbsp;&nbsp; Invoice Date: {mdY(inv_date)}",
            STYLES["Heading3"],
        ),
    ]
    if "conflicting_dates" in g.traps:
        service = c["_loss"] + timedelta(days=g.r.randint(0, 3))
        due = inv_date + timedelta(days=30)
        story.append(Paragraph(f"Service Date: {mdY(service)} &nbsp;&nbsp; Due Date: {mdY(due)}", SMALL))
    story += [
        *claim_reference(c, g),
        Paragraph("<b>Bill To:</b>", SMALL),
        kv_table(header_rows(c, g)),
        Spacer(1, 10),
    ]
    rows = [["Description", "Qty", "Unit", "Rate", "Amount"]]
    for li in items:
        rows.append(
            [
                Paragraph(escape(li["description"]), SMALL),
                f"{li['quantity']:g}",
                li["unit"],
                money(li["unit_price"]),
                money(li["amount"]),
            ]
        )
    t = Table(rows, colWidths=[w * inch for w in (3.6, 0.6, 0.5, 1.0, 1.1)])
    t.setStyle(table_style())
    totals = [
        ["Subtotal", money(subtotal)],
        ["Tax", money(tax)],
        ["Total", money(total)],
    ]
    if deposit:
        totals.append(["Less deposit received", money(-deposit)])
    totals.append([g.label("amount_due", "AMOUNT DUE"), money(amount_due)])
    tt = Table(totals, colWidths=[1.8 * inch, 1.3 * inch], hAlign="RIGHT")
    tt.setStyle(
        TableStyle(
            [
                ("ALIGN", (1, 0), (1, -1), "RIGHT"),
                ("FONT", (0, -1), (-1, -1), "Helvetica-Bold"),
            ]
        )
    )
    story += [
        t,
        Spacer(1, 10),
        tt,
        Spacer(1, 14),
        Paragraph("Payment due within 30 days. Thank you for your business.", SMALL),
    ]

    stamp_cb = None
    if stamp:
        text = "PAID" if g.r.random() < 0.5 else "RECEIVED"
        stamp_cb = stamp_painter(text, inv_date + timedelta(days=g.r.randint(2, 25)), g.r)
    fields = {k: c[k] for k in c if not k.startswith("_")}
    fields.update(
        vendor_name=vendor,
        invoice_number=inv_no,
        invoice_date=inv_date,
        subtotal=subtotal,
        tax=tax,
        total=total,
        amount_due=amount_due,
    )
    return build_pdf(story, stamp_cb), truth(DocType.invoice, fields, items, g.traps)


def fire_report(g: Gen) -> tuple[bytes, dict]:
    c = g.common()
    c["cause_of_loss"] = "fire"
    city = g.fake.city()
    dept = f"{city} Fire Department"
    incident_no = f"{c['_loss'].year}-{g.r.randint(10000, 99999)}"
    alarm = datetime.combine(c["_loss"], datetime.min.time()) + timedelta(minutes=g.r.randint(0, 24 * 60 - 1))
    arrival = alarm + timedelta(minutes=g.r.randint(4, 14))
    origin, cause_det = g.r.choice(FIRE_CAUSES)
    c["cause_of_loss_detail"] = cause_det
    injuries = g.r.choice([0, 0, 0, 1, 2])
    fatalities = 0
    narrative = (
        f"Engine and ladder companies responded to a reported structure fire at the above address. "
        f"On arrival crews found smoke showing from the {origin.lower()}. Fire was contained to the area "
        f"of origin and extinguished with one handline. Occupants were evacuated prior to arrival. "
        f"{'Civilian injuries were treated and transported by EMS. ' if injuries else ''}"
        f"Scene turned over to the owner after overhaul."
    )
    time_distractors = g.trap("time_distractors", 0.7)
    conflicting = g.trap("conflicting_dates", 0.7)
    hrs = g.trap("time_format", 0.5)
    g.trap("label_variants", 0.5)

    def t(dt: datetime) -> str:
        return dt.strftime("%H%M") + " hrs" if hrs else dt.strftime("%H:%M")

    rows = [("Incident Number", incident_no), ("Incident Date", mdY(c["_loss"]))]
    if conflicting:
        rows.append(("Report Date", mdY(c["_loss"] + timedelta(days=g.r.randint(1, 6)))))
    rows.append(("Alarm Time", t(alarm)))
    if time_distractors:
        rows.append(("Dispatch Time", t(alarm + timedelta(minutes=1))))
    rows.append(("Arrival Time", t(arrival)))
    if time_distractors:
        rows.append(("Controlled", t(arrival + timedelta(minutes=g.r.randint(8, 30)))))
        rows.append(("Last Unit Cleared", t(arrival + timedelta(minutes=g.r.randint(60, 180)))))
    rows += [
        ("Location", c["property_address"]),
        (g.label("owner", "Owner / Occupant"), c["insured_name"]),
        ("Area of Origin", origin),
        ("Cause Determination", cause_det),
        ("Civilian Injuries", str(injuries)),
        ("Civilian Fatalities", str(fatalities)),
    ]
    story = [
        Paragraph(f"<b>{dept.upper()}</b>", STYLES["Title"]),
        Paragraph("<b>INCIDENT REPORT</b> - Structure Fire", STYLES["Heading3"]),
        kv_table(rows),
        Spacer(1, 10),
        Paragraph("<b>Narrative</b>", STYLES["Heading4"]),
        Paragraph(narrative, SMALL),
        Spacer(1, 18),
        Paragraph(
            f"Reporting Officer: Capt. {g.fake.last_name()} &nbsp;&nbsp; Badge {g.r.randint(100, 999)}",
            SMALL,
        ),
    ]
    fields = {
        "claim_number": None,
        "policy_number": None,
        "insured_name": c["insured_name"],
        "property_address": c["property_address"],
        "date_of_loss": c["_loss"],
        "cause_of_loss": "fire",
        "cause_of_loss_detail": cause_det,
        "incident_number": incident_no,
        "fire_department": dept,
        "alarm_time": alarm,
        "arrival_time": arrival,
        "area_of_origin": origin,
        "cause_determination": cause_det,
        "injuries": injuries,
        "fatalities": fatalities,
        "narrative_summary": narrative,
    }
    return build_pdf(story), truth(DocType.fire_report, fields, [], g.traps)


def truth(doc_type: DocType, fields: dict, items: list[dict], traps: list[str]) -> dict:
    def fmt(v):
        if isinstance(v, datetime):
            return v.strftime("%Y-%m-%dT%H:%M")
        if isinstance(v, date):
            return v.isoformat()
        return v

    expected = [f.value for f in EXPECTED_FIELDS[doc_type]]
    missing = set(expected) - set(fields)
    assert not missing, f"generator forgot fields: {missing}"
    out = {
        "doc_type": doc_type.value,
        "fields": {k: fmt(fields[k]) for k in expected},
        "line_items": [{k: v for k, v in li.items() if not k.startswith("_")} for li in items],
    }
    if traps:
        out["traps"] = list(traps)
    return out


def scanify(pdf_bytes: bytes, r: random.Random, hard: bool = False) -> bytes:
    """Rasterize and degrade a PDF so it looks like a scan with no text layer. The hard
    profile scans at lower resolution with more skew, noise, blur and compression."""
    import pypdfium2 as pdfium

    dpi = 110 if hard else 150
    skew = 3.0 if hard else 1.5
    pages = []
    for page in pdfium.PdfDocument(pdf_bytes):
        img = page.render(scale=dpi / 72).to_pil().convert("L")
        if hard:
            img = img.point(lambda p: int(45 + p * 0.72))  # washed-out contrast
        img = img.rotate(r.uniform(-skew, skew), resample=Image.BICUBIC, expand=False, fillcolor=255)
        noise = Image.effect_noise(img.size, 30 if hard else 18).point(lambda p: p - 128)
        img = Image.blend(img, Image.merge("L", [noise]).convert("L"), 0.10 if hard else 0.06)
        img = img.filter(ImageFilter.GaussianBlur(0.8 if hard else 0.5))
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=r.randint(25, 40) if hard else r.randint(45, 70))
        pages.append(Image.open(io.BytesIO(buf.getvalue())))
    out = io.BytesIO()
    pages[0].save(out, "PDF", save_all=True, append_images=pages[1:], resolution=dpi)
    return out.getvalue()


GENERATORS = {
    "estimate": contractor_estimate,
    "invoice": invoice,
    "fire": fire_report,
}
PROFILES = {
    "easy": {"seed": 7, "scan_ratio": 1 / 3, "out": "samples/synthetic"},
    "hard": {"seed": 11, "scan_ratio": 0.4, "out": "samples/synthetic-hard"},
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--profile", choices=PROFILES, default="easy")
    ap.add_argument("--per-type", type=int, default=10)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--scan-ratio", type=float)
    ap.add_argument("--out")
    args = ap.parse_args()
    prof = PROFILES[args.profile]
    seed = args.seed if args.seed is not None else prof["seed"]
    scan_ratio = args.scan_ratio if args.scan_ratio is not None else prof["scan_ratio"]
    hard = args.profile == "hard"

    out = Path(args.out or prof["out"])
    out.mkdir(parents=True, exist_ok=True)
    g = Gen(seed, hard=hard)
    for kind, fn in GENERATORS.items():
        for i in range(1, args.per_type + 1):
            g.traps = []
            pdf, t = fn(g)
            scan = g.r.random() < scan_ratio
            if scan:
                pdf = scanify(pdf, g.r, hard=hard)
            t["scan"] = scan
            doc_id = f"{kind}-{i:02d}{'-scan' if scan else ''}"
            (out / f"{doc_id}.pdf").write_bytes(pdf)
            (out / f"{doc_id}.truth.json").write_text(json.dumps(t, indent=2))
            print(f"{doc_id}: {len(t['line_items'])} line items  traps={','.join(t.get('traps', [])) or '-'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
