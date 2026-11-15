"""Export a Report to Word, Excel, PDF, HTML, CSV, images and a JSON analysis record."""
from __future__ import annotations

import base64
import datetime as _dt
import html
import io
import json
import math
import re
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .report import Report

FORMATS = {
    "docx": "Word document (.docx)", "xlsx": "Excel workbook (.xlsx)", "pdf": "PDF report (.pdf)",
    "html": "Web page (.html, self-contained)", "csv": "CSV tables (.zip)",
    "png": "Figures as PNG, 300 dpi (.zip)", "tiff": "Figures as TIFF, 600 dpi for journals (.zip)",
    "svg": "Figures as SVG vector (.zip)", "json": "Analysis record for reproducibility (.json)",
}


def _fmt_cell(v: Any, col: str = "") -> Any:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return ""
    if isinstance(v, (bool, np.bool_)):
        return "Yes" if v else "No"
    if isinstance(v, (float, np.floating)):
        c = str(col).lower()
        if c in ("p", "q p", "p_between", "q_p", "qe_p", "qm_p") or c.endswith(" p") or c.startswith("p "):
            return "<0.001" if v < 0.001 else f"{v:.3f}"
        if float(v).is_integer() and abs(v) < 1e9:
            return f"{int(v)}"
        return f"{v:.3f}" if abs(v) < 1000 else f"{v:.1f}"
    return v


def display_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in out.columns:
        out[c] = [_fmt_cell(v, c) for v in out[c]]
    return out.astype(object).where(pd.notna(out), "")


def _fig_bytes(fig, fmt="png", dpi=300) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches="tight")
    return buf.getvalue()


def _safe(name: str, n=60) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", name).strip("_")[:n] or "item"


def export(report: Report, formats: list[str], out_dir: str | Path, stem: str = "MetaPy_report",
           record: dict[str, Any] | None = None) -> list[str]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for f in formats:
        fn = {"docx": _docx, "xlsx": _xlsx, "pdf": _pdf, "html": _html, "csv": _csv,
              "png": lambda r, p: _figzip(r, p, "png", 300), "tiff": lambda r, p: _figzip(r, p, "tiff", 600),
              "svg": lambda r, p: _figzip(r, p, "svg", 300)}.get(f)
        if f == "json":
            p = out_dir / f"{stem}_analysis_record.json"
            p.write_text(json.dumps(record or {}, indent=2, default=str))
            paths.append(str(p))
            continue
        if fn is None:
            continue
        ext = {"csv": "_tables.zip", "png": "_figures_png.zip", "tiff": "_figures_tiff.zip", "svg": "_figures_svg.zip"}.get(f, f".{f}")
        p = out_dir / f"{stem}{ext}"
        fn(report, p)
        paths.append(str(p))
    return paths


# ------------------------------------------------------------------ Word
def _docx(rep: Report, path: Path):
    from docx import Document
    from docx.enum.section import WD_ORIENT
    from docx.shared import Inches, Pt, RGBColor

    doc = Document()
    st = doc.styles["Normal"]
    st.font.name = "Calibri"
    st.font.size = Pt(10.5)
    doc.add_heading(rep.title, 0)
    p = doc.add_paragraph(f"Generated {_dt.date.today().isoformat()} by MetaPy. ")
    meta = doc.add_table(rows=0, cols=2)
    meta.style = "Light List Accent 1"
    for k, v in rep.meta.items():
        row = meta.add_row().cells
        row[0].text, row[1].text = str(k), str(v)
    for sec in rep.sections:
        _orient(doc, WD_ORIENT.PORTRAIT)
        doc.add_heading(sec.heading, 1)
        for t in sec.text:
            doc.add_paragraph(t)
        for name, fig in sec.figures:
            doc.add_heading(name, 3)
            doc.add_picture(io.BytesIO(_fig_bytes(fig, "png", 300)), width=Inches(6.3))
        for name, df in sec.tables:
            d = display_frame(df)
            wide = len(d.columns) > 6
            _orient(doc, WD_ORIENT.LANDSCAPE if wide else WD_ORIENT.PORTRAIT)
            doc.add_heading(name, 3)
            if d.empty:
                doc.add_paragraph("No rows.")
                continue
            table = doc.add_table(rows=1, cols=len(d.columns))
            table.style = "Light Grid Accent 1"
            for i, c in enumerate(d.columns):
                cell = table.rows[0].cells[i]
                cell.text = str(c)
            for _, r in d.iterrows():
                cells = table.add_row().cells
                for i, v in enumerate(r):
                    cells[i].text = str(v)
            size = Pt(8 if len(d.columns) <= 8 else 7)
            for row in table.rows:
                for cell in row.cells:
                    for par in cell.paragraphs:
                        for run in par.runs:
                            run.font.size = size

    doc.add_paragraph().add_run("Values calculated by MetaPy are listed with formulas in the derivation log. "
                                "MetaPy checks internal consistency only; it does not verify data against the original publications.").italic = True
    doc.save(path)


def _orient(doc, orientation):
    """Start a new section only when the page orientation actually changes."""
    cur = doc.sections[-1]
    if cur.orientation == orientation:
        return
    new = doc.add_section()
    new.orientation = orientation
    if (orientation == 1 and new.page_width < new.page_height) or (orientation == 0 and new.page_width > new.page_height):
        new.page_width, new.page_height = new.page_height, new.page_width


# ------------------------------------------------------------------ Excel
def _xlsx(rep: Report, path: Path):
    from openpyxl.drawing.image import Image as XLImage
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    used = set()

    def sheet_name(n):
        base = re.sub(r"[\[\]\*\?/\\:]", "", n)[:28] or "Sheet"
        name, i = base, 2
        while name.lower() in used:
            name = f"{base[:25]}_{i}"
            i += 1
        used.add(name.lower())
        return name

    with pd.ExcelWriter(path, engine="openpyxl") as xw:
        info = pd.DataFrame(list(rep.meta.items()), columns=["Item", "Value"])
        texts = [(sec.heading, t) for sec in rep.sections for t in sec.text]
        if texts:
            info = pd.concat([info, pd.DataFrame([{"Item": h, "Value": t} for h, t in texts])], ignore_index=True)
        info.to_excel(xw, sheet_name=sheet_name("Report"), index=False)
        for sec in rep.sections:
            for name, df in sec.tables:
                sn = sheet_name(f"{sec.heading[:12]} {name}" if len(rep.sections) > 2 else name)
                df.to_excel(xw, sheet_name=sn, index=False)
        wb = xw.book
        for ws in wb.worksheets:
            for c in ws[1]:
                c.font = Font(bold=True, color="FFFFFF")
                c.fill = PatternFill("solid", fgColor="1F3A5F")
                c.alignment = Alignment(wrap_text=True, vertical="top")
            for i, col in enumerate(ws.columns, 1):
                width = max((len(str(c.value)) for c in col if c.value is not None), default=8)
                ws.column_dimensions[get_column_letter(i)].width = min(60, max(10, width + 2))
            ws.freeze_panes = "A2"
        figs = [(sec.heading, n, f) for sec in rep.sections for n, f in sec.figures]
        if figs:
            ws = wb.create_sheet("Figures")
            row = 1
            for h, n, f in figs:
                ws.cell(row=row, column=1, value=f"{h} — {n}").font = Font(bold=True)
                img = XLImage(io.BytesIO(_fig_bytes(f, "png", 110)))
                ws.add_image(img, f"A{row + 1}")
                row += int(img.height / 20) + 4


# ------------------------------------------------------------------ PDF
def _pdf(rep: Report, path: Path):
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.lib.units import cm
    from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

    styles = getSampleStyleSheet()
    small = styles["BodyText"].clone("small", fontSize=7, leading=8.5)
    doc = SimpleDocTemplate(str(path), pagesize=landscape(A4), leftMargin=1.5 * cm, rightMargin=1.5 * cm, topMargin=1.5 * cm, bottomMargin=1.5 * cm)
    W = landscape(A4)[0] - 3 * cm
    story = [Paragraph(html.escape(rep.title), styles["Title"])]
    for k, v in rep.meta.items():
        story.append(Paragraph(f"<b>{html.escape(str(k))}:</b> {html.escape(str(v))}", styles["BodyText"]))
    for sec in rep.sections:
        story += [Spacer(1, 10), Paragraph(html.escape(sec.heading), styles["Heading1"])]
        for t in sec.text:
            story.append(Paragraph(html.escape(t), styles["BodyText"]))
        for name, fig in sec.figures:
            png = _fig_bytes(fig, "png", 200)
            img = Image(io.BytesIO(png))
            ratio = img.imageHeight / img.imageWidth
            w = min(W * 0.75, 22 * cm)
            h = w * ratio
            if h > 17 * cm:
                h = 17 * cm
                w = h / ratio
            img.drawWidth, img.drawHeight = w, h
            story += [Paragraph(html.escape(name), styles["Heading3"]), img]
        for name, df in sec.tables:
            d = display_frame(df)
            story.append(Paragraph(html.escape(name), styles["Heading3"]))
            if d.empty:
                story.append(Paragraph("No rows.", styles["BodyText"]))
                continue
            data = [[Paragraph(f"<b>{html.escape(str(c))}</b>", small) for c in d.columns]]
            data += [[Paragraph(html.escape(str(v)), small) for v in r] for r in d.itertuples(index=False)]
            t = Table(data, repeatRows=1, colWidths=[W / len(d.columns)] * len(d.columns))
            t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#DCE6F0")),
                                   ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#999999")),
                                   ("VALIGN", (0, 0), (-1, -1), "TOP")]))
            story.append(t)
    doc.build(story)


# ------------------------------------------------------------------ HTML
def _html(rep: Report, path: Path):
    parts = [f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>{html.escape(rep.title)}</title>",
             "<style>body{font-family:Segoe UI,Arial,sans-serif;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#222}"
             "table{border-collapse:collapse;font-size:12px;margin:0.5rem 0 1.5rem}td,th{border:1px solid #ccc;padding:3px 6px;vertical-align:top}"
             "th{background:#1F3A5F;color:#fff}img{max-width:100%}h1{color:#1F3A5F}</style></head><body>",
             f"<h1>{html.escape(rep.title)}</h1><ul>"]
    parts += [f"<li><b>{html.escape(str(k))}:</b> {html.escape(str(v))}</li>" for k, v in rep.meta.items()]
    parts.append("</ul>")
    for sec in rep.sections:
        parts.append(f"<h2>{html.escape(sec.heading)}</h2>")
        parts += [f"<p>{html.escape(t)}</p>" for t in sec.text]
        for name, fig in sec.figures:
            b64 = base64.b64encode(_fig_bytes(fig, "png", 150)).decode()
            parts.append(f"<h3>{html.escape(name)}</h3><img alt='{html.escape(name)}' src='data:image/png;base64,{b64}'>")
        for name, df in sec.tables:
            parts.append(f"<h3>{html.escape(name)}</h3>" + display_frame(df).to_html(index=False, escape=True))
    parts.append("</body></html>")
    path.write_text("".join(parts), encoding="utf-8")


def _csv(rep: Report, path: Path):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        i = 1
        for sec in rep.sections:
            for name, df in sec.tables:
                z.writestr(f"{i:02d}_{_safe(sec.heading, 30)}_{_safe(name, 40)}.csv", df.to_csv(index=False))
                i += 1


def _figzip(rep: Report, path: Path, fmt: str, dpi: int):
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        i = 1
        for sec in rep.sections:
            for name, fig in sec.figures:
                if fmt == "tiff":
                    from PIL import Image as PILImage
                    im = PILImage.open(io.BytesIO(_fig_bytes(fig, "png", dpi)))
                    buf = io.BytesIO()
                    im.convert("RGB").save(buf, format="TIFF", compression="tiff_lzw", dpi=(dpi, dpi))
                    data = buf.getvalue()
                else:
                    data = _fig_bytes(fig, fmt, dpi)
                z.writestr(f"{i:02d}_{_safe(sec.heading, 30)}_{_safe(name, 40)}.{('tif' if fmt == 'tiff' else fmt)}", data)
                i += 1
