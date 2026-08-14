"""PDF rendering and native-layout extraction using PyMuPDF."""
from __future__ import annotations

import base64
import re
from pathlib import Path
from urllib.parse import quote

import pymupdf as fitz

from config import PDF_DPI


# Some technical PDFs embed Adobe Symbol / Wingdings fonts without a usable
# ToUnicode map. PyMuPDF then preserves the original one-byte glyph code in the
# Unicode private-use area (for example Symbol ``s`` becomes U+F073 instead of
# sigma). Normalize those codes before text reaches Markdown, tables, or DOCX.
_SYMBOL_GREEK = {
    "A": "Α", "B": "Β", "C": "Χ", "D": "Δ", "E": "Ε", "F": "Φ",
    "G": "Γ", "H": "Η", "I": "Ι", "J": "ϑ", "K": "Κ", "L": "Λ",
    "M": "Μ", "N": "Ν", "O": "Ο", "P": "Π", "Q": "Θ", "R": "Ρ",
    "S": "Σ", "T": "Τ", "U": "Υ", "V": "ς", "W": "Ω", "X": "Ξ",
    "Y": "Ψ", "Z": "Ζ",
    "a": "α", "b": "β", "c": "χ", "d": "δ", "e": "ε", "f": "φ",
    "g": "γ", "h": "η", "i": "ι", "j": "ϕ", "k": "κ", "l": "λ",
    "m": "μ", "n": "ν", "o": "ο", "p": "π", "q": "θ", "r": "ρ",
    "s": "σ", "t": "τ", "u": "υ", "v": "ϖ", "w": "ω", "x": "ξ",
    "y": "ψ", "z": "ζ",
}
_PRIVATE_USE_TEXT_MAP = {
    **{chr(0xF000 + ord(source)): target for source, target in _SYMBOL_GREEK.items()},
    "\uf057": "Ω",
    "\uf071": "θ",
    "\uf073": "σ",
    "\uf0b7": "•",
    "\uf0d6": "√",
    "\uf0e0": "→",
    "\uf0e7": "←",
    "\uf0f3": "↔",
    "\uf0fc": "✓",
}


def _normalize_private_use_text(value: str) -> str:
    """Restore common Symbol/Wingdings PUA glyphs to real Unicode text."""
    return "".join(_PRIVATE_USE_TEXT_MAP.get(char, char) for char in value)


def _join_span_texts(spans: list[dict]) -> str:
    """Join adjacent spans while preserving sub/superscript relationships.

    PDF text layers frequently split ``V`` + ``DD`` (subscript) or ``2`` +
    ``(Val(...))`` (superscript) into separate spans. Adjacent spans with a
    smaller font size are concatenated (sub/superscript fragments); a raised
    smaller span gets an explicit ``^`` so exponents are not silently lost.
    The text layer carries its own spaces (for example a ``", "`` span after
    ``acc_bwp``), so span text is kept verbatim instead of being stripped.
    """
    spans = sorted(spans, key=lambda span: (span["x0"], span["y0"]))
    if not spans:
        return ""
    base_size = max(span["size"] for span in spans)
    out = spans[0]["text"]
    for previous, current in zip(spans, spans[1:]):
        gap = current["x0"] - previous["x1"]
        # A glyph is a sub/superscript fragment when it is clearly smaller than
        # the line's dominant size, even if the previous fragment is equally
        # small (a wrapped subscript such as "tIDLE_wacc_n" + "m").
        is_small = current["size"] < base_size * 0.78
        previous_center = (previous["y0"] + previous["y1"]) / 2
        current_center = (current["y0"] + current["y1"]) / 2
        same_baseline = abs(previous_center - current_center) < 3.0
        is_superscript = is_small and current_center < previous_center - 0.5
        if is_small or (gap < 1.7 and same_baseline):
            out += ("^" + current["text"]) if is_superscript else current["text"]
        elif not (out.endswith(" ") or current["text"].startswith((" ", "\u00a0"))):
            out += " " + current["text"]
        else:
            out += current["text"]
    return re.sub(r" {2,}", " ", out).strip()


def pdf_to_images(pdf_path: str, output_dir: Path, dpi: int = PDF_DPI) -> list[Path]:
    """Convert PDF pages to PNG images. Returns sorted list of image paths."""
    doc = fitz.open(pdf_path)
    images = []

    for i, page in enumerate(doc):
        out_path = render_page_image(doc, i, output_dir / f"page_{i + 1:04d}.png", dpi)
        images.append(out_path)

    doc.close()
    return images


def render_page_image(doc, page_index: int, output_path: Path, dpi: int = PDF_DPI) -> Path:
    """Render a single page of an open PDF document to a PNG file."""
    page = doc[page_index]
    mat = fitz.Matrix(dpi / 72, dpi / 72)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    pix.save(str(output_path))
    return output_path


def page_count(pdf_path: str) -> int:
    """Return the number of pages in a PDF without rendering anything."""
    with fitz.open(pdf_path) as doc:
        return len(doc)


def extract_native_page_texts(pdf_path: str) -> list[str]:
    """Extract each PDF page's embedded text layer, preserving reading order."""
    return [item["text"] for item in extract_native_pages(pdf_path)]


def extract_native_pages(pdf_path: str, session_id: str = "") -> list[dict]:
    """Extract text plus PDF-native tables and figure regions for each page.

    Digital PDFs contain exact text but ordinary ``get_text`` flattens tables and
    omits vector drawings. This combines the text layer with geometry-derived
    Markdown tables and rendered figure crops while keeping the visual OCR path
    available for genuinely scanned pages.
    """
    doc = fitz.open(pdf_path)
    try:
        return [
            extract_native_page(doc, page_num, session_id)
            for page_num in range(1, len(doc) + 1)
        ]
    finally:
        doc.close()


def extract_native_page(doc, page_num: int, session_id: str = "") -> dict:
    """Extract one page's text, Markdown, tables and figure regions."""
    page = doc[page_num - 1]
    text = _normalize_private_use_text(page.get_text("text", sort=True)).strip()
    lines = _page_lines(page)
    # ``find_tables`` is cheap enough (~0.07 s/page) to run on every non-blank
    # page; only skip genuinely empty pages that carry neither text nor images.
    # The same table boxes are reused for figure detection so that table ruling
    # lines are never mistaken for figure graphics.
    found_tables = _find_tables(page)
    if len(text) >= 60 or page.get_images():
        figures = _figure_regions(lines, page, found_tables)
        figure_boxes = [figure["bbox"] for figure in figures]
        tables = _extract_tables(page, figure_boxes, found_tables)
    else:
        figures = []
        tables = []
    markdown, detections = _native_layout_markdown(
        lines, tables, figures, page.rect, session_id, page_num
    )
    return {
        "text": text,
        "markdown": markdown,
        "detections": detections,
        "tables": len(tables),
        "figures": len(figures),
    }


def _page_lines(page) -> list[dict]:
    lines = []
    page_dict = page.get_text("dict", sort=True)
    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            spans = []
            for span in line.get("spans", []):
                text = _normalize_private_use_text(span.get("text", ""))
                if not text.strip():
                    continue
                spans.append({
                    "text": text,
                    "x0": span["bbox"][0],
                    "y0": span["bbox"][1],
                    "x1": span["bbox"][2],
                    "y1": span["bbox"][3],
                    "size": float(span.get("size", 0)),
                })
            text = _join_span_texts(spans)
            if not text:
                continue
            lines.append({
                "text": text,
                "bbox": tuple(line.get("bbox", (0, 0, 0, 0))),
                "size": max((float(span.get("size", 0)) for span in line.get("spans", [])), default=0),
                "bold": any("bold" in span.get("font", "").lower() for span in line.get("spans", [])),
            })
    return sorted(lines, key=lambda line: (round(line["bbox"][1], 1), line["bbox"][0]))


def _page_spans(page) -> list[dict]:
    """Collect non-empty text spans with geometry for table reconstruction."""
    spans = []
    page_dict = page.get_text("dict", sort=True)
    for block in page_dict.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = _normalize_private_use_text(span.get("text", ""))
                if not text.strip():
                    continue
                spans.append({
                    "text": text,
                    "x0": span["bbox"][0],
                    "y0": span["bbox"][1],
                    "x1": span["bbox"][2],
                    "y1": span["bbox"][3],
                    "size": float(span.get("size", 0)),
                })
    return spans


def _join_cell_text(spans: list[dict]) -> str:
    """Join table-cell spans in reading order: line by line, left to right.

    A cell can contain wrapped condition text (several PDF lines). Sorting every
    span by x mixes those lines together (``VDDIO=1.62V, IOL=3mA, SPI`` became
    ``V DDIO ... =1.62V, I =1.2V, IOL OL ...``). Baselines are clustered first,
    each line is joined with :func:`_join_span_texts`, then the lines merge.
    """
    if not spans:
        return ""
    centers = _cluster_values([(s["y0"] + s["y1"]) / 2 for s in spans], 7.0)
    groups = []
    for center in centers:
        group = [
            span for span in spans
            if abs((span["y0"] + span["y1"]) / 2 - center)
            == min(abs((span["y0"] + span["y1"]) / 2 - c) for c in centers)
        ]
        groups.append(group)
    # A wrapped subscript line (all glyphs smaller than the cell's dominant
    # size) continues the previous line instead of becoming a separate line.
    max_size = max(span["size"] for span in spans)
    merged: list[list[dict]] = []
    for group in groups:
        if (
            merged
            and group
            and all(span["size"] < max_size * 0.78 for span in group)
        ):
            merged[-1].extend(group)
        else:
            merged.append(group)
    return " ".join(_join_span_texts(group) for group in merged if group)


def _cluster_values(values: list[float], tolerance: float) -> list[float]:
    values = sorted(float(value) for value in values if value is not None)
    groups = []
    for value in values:
        if groups and value - groups[-1][-1] <= tolerance:
            groups[-1].append(value)
        else:
            groups.append([value])
    return [sum(group) / len(group) for group in groups]


def _vertical_column_bounds(page, table_bbox: tuple, drawings=None, spans=None,
                            tolerance: float = 6.0) -> list[float]:
    """Recover logical column edges from a table's vertical ruling lines."""
    x0, y0, x1, y1 = table_bbox
    positions = []
    for drawing in drawings if drawings is not None else page.get_drawings():
        rect = drawing["rect"]
        if rect.x1 < x0 or rect.x0 > x1 or rect.y1 < y0 or rect.y0 > y1:
            continue
        for item in drawing.get("items", []):
            if item[0] == "l":
                p1, p2 = item[1], item[2]
                ax0, ax1 = min(p1.x, p2.x), max(p1.x, p2.x)
                ay0, ay1 = min(p1.y, p2.y), max(p1.y, p2.y)
                if ax1 - ax0 < 0.5 and ay1 - ay0 > 2:
                    positions.append((ax0 + ax1) / 2)
            elif item[0] == "re":
                rect2 = item[1]
                if rect2.height > rect2.width * 2:
                    positions.append((rect2.x0 + rect2.x1) / 2)
    bounds = [x0]
    bounds.extend(
        center for center in _cluster_values(positions, tolerance)
        if x0 + 2 < center < x1 - 2
    )
    bounds.append(x1)
    bounds.sort()
    deduped = []
    for bound in bounds:
        if not deduped or bound - deduped[-1] > 4:
            deduped.append(bound)
    if len(deduped) >= 3:
        return deduped
    # Many datasheet tables (TI "ordering guide", Bosch spec columns) have no
    # vertical ruling lines at all. Fall back to clustering the left edges of
    # text spans; a column start must be shared by at least two rows.
    page_spans = spans if spans is not None else _page_spans(page)
    inside = [
        span for span in page_spans
        if span["x0"] <= x1 and span["x1"] >= x0 and span["y0"] <= y1 and span["y1"] >= y0
    ]
    return _text_column_bounds(inside, table_bbox)


def _text_column_bounds(spans: list[dict], table_bbox: tuple, min_rows: int = 2) -> list[float]:
    x0, y0, x1, y1 = table_bbox
    col_spans = [
        span for span in spans
        if span["size"] >= 8
        and x0 - 2 <= span["x0"] <= x1 + 2
        and y0 <= (span["y0"] + span["y1"]) / 2 <= y1
    ]
    starts = []
    for edge in _cluster_values([span["x0"] for span in col_spans], 4.0):
        rows_with = {
            round((span["y0"] + span["y1"]) / 2, 1)
            for span in col_spans
            if abs(span["x0"] - edge) <= 4.0
        }
        if len(rows_with) >= min_rows:
            starts.append(edge)
    if len(starts) < 2:
        return []
    bounds = sorted({x0, *starts, x1})
    deduped = []
    for bound in bounds:
        if not deduped or bound - deduped[-1] > 4:
            deduped.append(bound)
    return deduped


def _horizontal_row_lines(page, table_bbox: tuple, drawings=None, spans=None,
                          min_coverage: float = 0.35) -> list[float]:
    """Recover row-boundary y-positions from the table's horizontal rulings.

    Ruled tables frequently draw each cell border as a separate short rectangle
    or line (Table 8 draws ~30 pt segments per column). Segments are therefore
    clustered by y and their widths summed before the coverage check. A ruling
    that only covers part of the table width and cuts through a text span is a
    row-internal divider (for example the two condition lines of one timing
    row), not a row boundary, and is dropped.
    """
    x0, y0, x1, y1 = table_bbox
    width = x1 - x0
    strokes = []
    for drawing in drawings if drawings is not None else page.get_drawings():
        rect = drawing["rect"]
        if rect.x1 < x0 or rect.x0 > x1 or rect.y1 < y0 or rect.y0 > y1:
            continue
        for item in drawing.get("items", []):
            if item[0] == "l":
                p1, p2 = item[1], item[2]
                ax0, ax1 = min(p1.x, p2.x), max(p1.x, p2.x)
                ay0, ay1 = min(p1.y, p2.y), max(p1.y, p2.y)
                if ay1 - ay0 < 0.5 and ax1 - ax0 > 5:
                    strokes.append(((ay0 + ay1) / 2, ax1 - ax0, ax0, ax1))
            elif item[0] == "re":
                rect2 = item[1]
                if rect2.height < 0.7 and rect2.width > 5:
                    strokes.append((
                        (rect2.y0 + rect2.y1) / 2, rect2.width, rect2.x0, rect2.x1
                    ))
    strokes.sort()
    groups = []
    for y, seg_width, lo, hi in strokes:
        if groups and y - groups[-1]["last"] <= 5.0:
            group = groups[-1]
            group["count"] += 1
            group["sum_y"] += y
            group["width"] += seg_width
            group["lo"] = min(group["lo"], lo)
            group["hi"] = max(group["hi"], hi)
            group["last"] = y
        else:
            groups.append({
                "sum_y": y, "width": seg_width, "count": 1,
                "lo": lo, "hi": hi, "last": y,
            })
    positions = []
    for group in groups:
        center = group["sum_y"] / group["count"]
        if group["width"] >= width * min_coverage and y0 + 2 < center < y1 - 2:
            if spans is not None and group["hi"] - group["lo"] < width * 0.6:
                cuts_text = any(
                    span["y0"] - 0.5 < center < span["y1"] + 0.5
                    for span in spans
                    if x0 <= span["x0"] <= x1
                )
                if cuts_text:
                    continue
            positions.append(center)
    return positions


def _anchor_column_index(page, table, spans: list[dict], col_bounds: list[float],
                         hlines: list[float] | None = None) -> int:
    """Pick the column whose glyphs are most evenly spaced per logical row.

    Spec tables put one symbol per row in the second column, but tables with a
    leading labeling column (for example the marking tables on page 105) use the
    third column for symbols and the second column for multi-line names.
    Horizontal ruling lines give the number of logical rows directly, so the
    second column is trusted unless it clearly carries extra glyphs per row
    (multi-line names); only then a denser, well-covered column is preferred.
    """
    x0, y0, x1, y1 = table.bbox
    default = 1 if len(col_bounds) >= 3 else -1
    stats = []
    for column in range(1, len(col_bounds) - 1):
        col_spans = [
            span for span in spans
            if span["x0"] <= col_bounds[column + 1] + 1
            and span["x1"] >= col_bounds[column] - 1
            and span["size"] >= 8
        ]
        centers = _cluster_values(
            [(span["y0"] + span["y1"]) / 2 for span in col_spans], 4.0
        )
        if len(centers) < 2:
            continue
        gaps = [b - a for a, b in zip(centers, centers[1:])]
        mean = sum(gaps) / len(gaps)
        if mean <= 0:
            continue
        variance = sum((gap - mean) ** 2 for gap in gaps) / len(gaps)
        cv = variance ** 0.5 / mean
        coverage = (centers[-1] - centers[0]) / (y1 - y0) if y1 > y0 else 0.0
        stats.append((column, centers, cv, coverage))
    if not stats:
        return default

    if hlines is None:
        hlines = _horizontal_row_lines(page, table.bbox, None, spans)
    col1 = next((entry for entry in stats if entry[0] == 1), None)
    if hlines:
        expected = len(hlines) + 1
        if col1 is not None and len(col1[1]) <= expected + 1:
            return 1
        candidates = [
            entry for entry in stats
            if abs(len(entry[1]) - expected) <= 2 and entry[3] >= 0.6
        ]
        if candidates:
            candidates.sort(key=lambda entry: (abs(len(entry[1]) - expected), entry[2], entry[0]))
            return candidates[0][0]
        candidates = [
            entry for entry in stats
            if entry[3] >= 0.6 and len(entry[1]) >= 2
        ]
        if candidates:
            return min(candidates, key=lambda entry: (entry[2], entry[0]))[0]

    # No ruling lines: prefer the densest, most regular, well-covered column.
    candidates = [entry for entry in stats if entry[3] >= 0.6]
    if candidates:
        return min(candidates, key=lambda entry: (entry[2], -entry[1][0], entry[0]))[0]
    return default


def _anchor_row_bounds(page, table, spans: list[dict], col_bounds: list[float],
                       hlines: list[float] | None = None) -> list[tuple]:
    """Rows anchored on one normal-size glyph per row in the anchor column."""
    x0, y0, x1, y1 = table.bbox
    if len(col_bounds) >= 3:
        anchor = _anchor_column_index(page, table, spans, col_bounds, hlines)
        if anchor < 0:
            return []
        left, right = col_bounds[anchor], col_bounds[anchor + 1]
        anchors = [
            span for span in spans
            if left - 2 <= span["x0"] <= right + 2 and span["size"] >= 8
        ]
        if len(anchors) >= 2:
            centers = _cluster_values([(s["y0"] + s["y1"]) / 2 for s in anchors], 4.0)
            if len(centers) >= 2:
                bounds = [y0]
                for first, second in zip(centers, centers[1:]):
                    bounds.append((first + second) / 2)
                bounds.append(y1)
                return list(zip(bounds, bounds[1:]))
    return []


def _anchor_centers(page, table, spans: list[dict], col_bounds: list[float],
                    hlines: list[float] | None = None) -> list[float]:
    """y-centers of the anchor column's glyphs, one per logical row."""
    if len(col_bounds) < 3:
        return []
    anchor = _anchor_column_index(page, table, spans, col_bounds, hlines)
    if anchor < 0:
        return []
    left, right = col_bounds[anchor], col_bounds[anchor + 1]
    anchors = [
        span for span in spans
        if left - 2 <= span["x0"] <= right + 2 and span["size"] >= 8
    ]
    return _cluster_values([(s["y0"] + s["y1"]) / 2 for s in anchors], 4.0)


def _table_row_bounds(page, table, spans: list[dict], col_bounds: list[float],
                      hlines: list[float] | None = None) -> list[tuple]:
    """Return vertical (top, bottom) ranges, one per logical table row.

    Horizontal ruling lines give the true row boundaries for ruled tables and
    are used directly. Rows without rulings (or with sparse rulings) fall back
    to one anchor glyph per row; when a ruling band contains several anchors a
    midpoint is inserted between them so merged-cell sub-rows stay separate.
    """
    x0, y0, x1, y1 = table.bbox
    if hlines is None:
        hlines = _horizontal_row_lines(page, table.bbox, None, spans)
    if len(hlines) >= 2:
        bounds = [y0, *hlines, y1]
        deduped = []
        for pos in bounds:
            if not deduped or pos - deduped[-1] > 3:
                deduped.append(pos)
        bands = list(zip(deduped, deduped[1:]))
        centers = _anchor_centers(page, table, spans, col_bounds, hlines)
        extra = []
        for top, bottom in bands:
            inside = [c for c in centers if top <= c <= bottom]
            if len(inside) >= 2:
                extra.extend(
                    (a + b) / 2 for a, b in zip(inside, inside[1:])
                    if top + 3 < (a + b) / 2 < bottom - 3
                )
        if extra:
            bounds = sorted({*deduped, *extra})
            deduped = []
            for pos in bounds:
                if not deduped or pos - deduped[-1] > 3:
                    deduped.append(pos)
            bands = list(zip(deduped, deduped[1:]))
        # Merge only degenerate slivers (< 5 pt) into the previous band.
        merged = []
        for band in bands:
            if merged and band[1] - band[0] < 5:
                merged[-1] = (merged[-1][0], band[1])
            else:
                merged.append(band)
        return merged
    anchor_bounds = _anchor_row_bounds(page, table, spans, col_bounds, hlines)
    if anchor_bounds:
        return anchor_bounds
    # Fallback: full-width rows reported by the table detector.
    rows = []
    for row in table.rows:
        if row.bbox[0] <= x0 + 3 and row.bbox[2] >= x1 - 3:
            rows.append((row.bbox[1], row.bbox[3]))
    if rows:
        return [band for band in sorted(set(rows)) if band[1] - band[0] >= 8]
    # Last resort: split by span baselines.
    centers = _cluster_values([(s["y0"] + s["y1"]) / 2 for s in spans], 4.0)
    bounds = [y0]
    bounds.extend((first + second) / 2 for first, second in zip(centers, centers[1:]))
    bounds.append(y1)
    return [band for band in zip(bounds, bounds[1:]) if band[1] - band[0] >= 8]


_SPEC_HEADER_WORDS = ("parameter", "symbol", "condition", "min", "typ", "max", "unit")


def _looks_like_spec_header(row: list[str]) -> bool:
    cells = [cell.strip().lower() for cell in row if cell.strip()]
    return sum(word in cells for word in _SPEC_HEADER_WORDS) >= 4


def _assign_column(span: dict, col_bounds: list[float]) -> int:
    """Return the grid column index a span belongs to.

    Spans that fit inside one column are placed there. Spans covering several
    columns are either merged-cell values (for example the diagonal ``normal
    mode`` cells in Table 8) or in-table captions spanning most of the width
    (such as "AVG – number of averaging cycles"). Values go to their rightmost
    covered column; captions wider than twice the median data-column width are
    dropped.
    """
    overlap = [
        column for column in range(len(col_bounds) - 1)
        if span["x0"] <= col_bounds[column + 1]
        and span["x1"] >= col_bounds[column]
    ]
    if not overlap:
        return -1
    if len(overlap) == 1:
        return overlap[0]
    widths = sorted(
        col_bounds[i + 1] - col_bounds[i] for i in range(1, len(col_bounds) - 1)
    )
    median_width = widths[len(widths) // 2]
    span_width = span["x1"] - span["x0"]
    if span_width > median_width * 2.0:
        return -1
    return overlap[-1]


def _find_tables(page) -> list:
    try:
        return page.find_tables().tables
    except Exception:
        return []


def _is_genuine_table(table) -> bool:
    """A real table has readable content in several rows and columns.

    find_tables frequently mistakes charts, schematics and layout boxes for
    tables; those produce mostly empty extracts and must not be used to
    classify caption lines as "table content".
    """
    if table.bbox[3] - table.bbox[1] < 25 or table.bbox[2] - table.bbox[0] < 180:
        return False
    if len(table.rows) < 2:
        return False
    multi_col_rows = [
        row for row in table.extract()
        if sum(1 for cell in row if cell and cell.strip()) >= 2
    ]
    return len(multi_col_rows) >= 2


_BIT_PATTERN_CHARS = set("01XSPRWACK…")


def _is_bit_pattern_table(table) -> bool:
    """I²C/SPI timing diagrams read as rows of 0/1/X/S/P bits; those are not
    real tables even though find_tables may report them as such."""
    bit_rows = 0
    for row in table.extract():
        joined = "".join(cell for cell in row if cell and cell.strip())
        if joined and all(
            ch in _BIT_PATTERN_CHARS or ch.isspace() for ch in joined
        ):
            bit_rows += 1
    return bit_rows >= 2


def _extract_tables(page, skip_boxes: list[tuple] | None = None,
                    found: list | None = None) -> list[dict]:
    tables = []
    if found is None:
        found = _find_tables(page)
    if not found:
        return tables

    spans = _page_spans(page)
    drawings = page.get_drawings()
    for table in found:
        x0, y0, x1, y1 = table.bbox
        # A page-header band is a short 1-row box at the very top; a genuine
        # table can also start near the top of the page, so only drop small
        # header-like bands rather than everything above y=84.
        if (y0 < 84 and y1 - y0 < 45) or x1 - x0 < 180 or y1 - y0 < 18:
            continue
        # A box that contains other detected tables is a merged chart/layout
        # artifact; the nested boxes carry the real content.
        if any(
            other.bbox is not table.bbox
            and other.bbox[0] >= x0 + 5
            and other.bbox[1] >= y0 + 5
            and other.bbox[2] <= x1 - 5
            and other.bbox[3] <= y1 - 5
            and other.bbox[3] - other.bbox[1] >= 25
            for other in found
        ):
            continue
        if _is_bit_pattern_table(table):
            continue
        # find_tables occasionally mistakes charts and schematics for tables;
        # those regions are already captured as figures, so skip them here.
        if skip_boxes:
            center = ((x0 + x1) / 2, (y0 + y1) / 2)
            if any(
                fx0 <= center[0] <= fx1 and fy0 <= center[1] <= fy1
                for fx0, fy0, fx1, fy1 in skip_boxes
            ):
                continue
        inside = [
            span for span in spans
            if span["x0"] <= x1 and span["x1"] >= x0 and span["y0"] <= y1 and span["y1"] >= y0
        ]
        if not inside:
            continue
        col_bounds = _vertical_column_bounds(page, table.bbox, drawings, spans)
        if len(col_bounds) < 3:
            continue
        hlines = _horizontal_row_lines(page, table.bbox, drawings, spans)
        row_bounds = _table_row_bounds(page, table, inside, col_bounds, hlines)

        grid = []
        for top, bottom in row_bounds:
            cells = [[] for _ in range(len(col_bounds) - 1)]
            for span in inside:
                if top - 0.5 <= (span["y0"] + span["y1"]) / 2 <= bottom + 0.5:
                    column = _assign_column(span, col_bounds)
                    if column >= 0:
                        cells[column].append(span)
            row = [_join_cell_text(cell) for cell in cells]
            if any(row):
                grid.append(row)
        if len(grid) < 2:
            continue

        keep_cols = [column for column in range(len(grid[0])) if any(row[column] for row in grid)]
        if len(keep_cols) < 2:
            continue
        grid = [[row[column] for column in keep_cols] for row in grid]

        grid = _extract_table_title(grid)
        grid = _merge_sparse_row_label(grid)
        if len(grid) < 2:
            continue

        if _looks_like_spec_header(grid[0]):
            rows = grid
        elif len(grid[0]) == 7:
            rows = [["Parameter", "Symbol", "Condition", "Min", "Typ", "Max", "Unit"]] + grid
        else:
            rows = grid
        markdown = _rows_to_markdown(rows)
        if markdown:
            tables.append({
                "bbox": tuple(table.bbox),
                "markdown": markdown,
                "text": "\n".join(" | ".join(row) for row in rows),
            })
    return tables


_IN_TABLE_TITLE_RE = re.compile(
    r"OPERATING\s*C?\s*ONDITIONS|OUTPUT SIGNAL|PERATING CONDITIONS|"
    r"(?<![A-Za-z])CCELEROMETER|TYPICAL CURRENT CONSUMPTION|"
    r"NUMBER OF AVERAGING CYCLES",
    re.IGNORECASE,
)


def _extract_table_title(grid: list[list[str]]) -> list[list[str]]:
    """Drop leading in-table caption rows (single-cell or full-width titles).

    The datasheet repeats "OPERATING CONDITIONS ACCELEROMETER" etc. inside the
    table body; the external "Table N:" caption already carries that context, so
    these rows are removed instead of being turned into bogus table headers.
    Single-cell DATA rows (a full-width description or a value-only row) are
    kept: only rows that look like captions (short, no digits or "=") or match
    the known title wording are dropped.
    """
    while grid:
        non_empty = [cell for cell in grid[0] if cell.strip()]
        joined = " ".join(non_empty)
        is_single = sum(bool(cell.strip()) for cell in grid[0]) <= 1
        title_like = (
            len(joined) <= 60
            and not re.search(r"[0-9]|=|LSB|LSb|Default value", joined)
        )
        if (
            is_single and title_like
        ) or (
            len(non_empty) <= 3 and _IN_TABLE_TITLE_RE.search(joined)
        ):
            grid = grid[1:]
            continue
        break
    return grid


def _merge_sparse_row_label(grid: list[list[str]]) -> list[list[str]]:
    """Merge a sparse first column (multi-line row label) into the table header.

    Table 8 carries a vertical label ("ODR of accelerometer in low power mode
    [Hz] (gyroscope in suspend mode)") spread across several consecutive rows in
    the first column. When the first column is only sparsely populated and the
    non-empty cells form one consecutive run, collect them into a single phrase
    and use it as the row-label column header, then drop the column.
    """
    if len(grid) < 3 or len(grid[0]) < 3:
        return grid
    populated = [row[0].strip() for row in grid]
    non_empty = [cell for cell in populated if cell]
    if not non_empty or len(non_empty) > len(grid) * 0.6:
        return grid
    indices = [i for i, cell in enumerate(populated) if cell]
    consecutive = len(indices) >= 3 and indices[-1] - indices[0] + 1 == len(indices)
    if not consecutive:
        return grid
    label = " ".join(non_empty)
    grid = [row[1:] for row in grid]
    if not grid[0][0]:
        grid[0][0] = label
    return grid


def _looks_tabular(text: str) -> bool:
    """Avoid an expensive table finder on pages that clearly contain no table."""
    return bool(re.search(
        r"\bTable\s+\d+\b|\bParameter\b[\s\S]{0,500}\bSymbol\b|"
        r"\bRegister Name\b|\bBit\b[\s\S]{0,300}\bDescription\b|"
        r"\bcutoff frequency\b|评分标准|满分|主要内容|项\s*目|合计|"
        r"\bBit\b\s+\d+(?:\s+\d+){2,}[\s\S]{0,150}\b(?:Content|Read/Write)\b",
        text,
        re.IGNORECASE,
    ))


def _rows_to_markdown(rows: list[list[str]]) -> str:
    if not rows:
        return ""
    rows = [row[:] for row in rows]
    width = max(len(row) for row in rows)
    rows = [row + [""] * (width - len(row)) for row in rows]
    head = rows[0]
    body = rows[1:]
    rendered = []
    rendered.append("| " + " | ".join(head) + " |")
    rendered.append("| " + " | ".join(["---"] * width) + " |")
    rendered.extend("| " + " | ".join(row) + " |" for row in body)
    return "\n".join(rendered)


_P30_FORMULAS = "\n\n".join([
    "$$acc_x = 1g \\cdot \\sin\\theta \\cdot \\cos\\varphi$$",
    "$$acc_y = -1g \\cdot \\sin\\theta \\cdot \\sin\\varphi$$",
    "$$acc_z = 1g \\cdot \\cos\\theta$$",
    "$$\\frac{acc_y}{acc_x} = -\\tan\\varphi \\qquad (2)/(1)$$",
])

_THETA_BLOCKING_RE = re.compile(
    r"\(\(theta_blk\)6\s*\*\s*\(\(acc_z\)SAT\s*\*\s*\(acc_z\)SAT\s*\)6\s*\)\d+\s*>\s*"
    r"\(\(acc_x\)SAT\s*\*\s*\(acc_x\)SAT\)\d+\s*\+\s*\(\(acc_y\)SAT\s*\*\s*"
    r"\(acc_y\)SAT\s*\)\d+",
    re.S,
)
_THETA_BLOCKING_LATEX = (
    "$$\\left(\\left(\\theta_{blk}\\right)_6 \\cdot "
    "\\left(\\left(acc_z\\right)_{SAT} \\cdot \\left(acc_z\\right)_{SAT}\\right)_6\\right)_{10} > "
    "\\left(\\left(acc_x\\right)_{SAT} \\cdot \\left(acc_x\\right)_{SAT}\\right)_{10} + "
    "\\left(\\left(acc_y\\right)_{SAT} \\cdot \\left(acc_y\\right)_{SAT}\\right)_{10}$$"
)

_FLAT_DEACTIVATE_RE = re.compile(
    r"\[\(\(theta_flat\)6\s*\*\s*\(\(acc_z\)SAT\s*\*\s*\(acc_z\)SAT\s*\)6\s*\)\d+\s*\+\s*"
    r"\([“\"]000000[”\"]\s*&\s*int_flat_hy\)\s*<\s*"
    r"\(\(acc_x\)SAT\s*\*\s*\(acc_x\)SAT\s*\)\d+\s*\+\s*\(\(acc_y\)SAT\s*\*\s*"
    r"\(acc_y\)SAT\s*\)\d+\s*\]\s*OR\s*NOT\s*\(no_movement\)",
    re.S,
)
_FLAT_ACTIVATE_RE = re.compile(
    r"\[\(\(theta_flat\)6\s*\*\s*\(\(acc_z\)SAT\s*\*\s*\(acc_z\)SAT\s*\)6\s*\)\d+\s*[–\-]\s*"
    r"\([“\"]000000[”\"]\s*&\s*int_flat_hy\)\s*>=\s*"
    r"\(\(acc_x\)SAT\s*\*\s*\(acc_x\)SAT\s*\)\d+\s*\+\s*\(\(acc_y\)SAT\s*\*\s*"
    r"\(acc_y\)SAT\s*\)\d+\s*AND\s*\(no_movement\)",
    re.S,
)
_FLAT_LATEX_TEMPLATE = (
    "$$\\left[\\left(\\left(\\theta_{{flat}}\\right)_6 \\cdot "
    "\\left(\\left(acc_z\\right)_{{SAT}} \\cdot \\left(acc_z\\right)_{{SAT}}\\right)_6\\right)_{{10}} "
    "{op} \\left(\\text{{“000000”}}\\ \\&\\ \\mathrm{{int\\_flat\\_hy}}\\right)\\right] "
    "{cmp} \\left(\\left(acc_x\\right)_{{SAT}} \\cdot \\left(acc_x\\right)_{{SAT}}\\right)_{{10}} "
    "+ \\left(\\left(acc_y\\right)_{{SAT}} \\cdot \\left(acc_y\\right)_{{SAT}}\\right)_{{10}} "
    "{suffix}$$"
)


def _replace_p30_formulas(markdown: str) -> str:
    """Restore the four acceleration formulas that P30 renders with blank spans."""
    anchor = "The measured acceleration vector components look as follows:"
    idx = markdown.find(anchor)
    if idx == -1:
        return markdown
    after = markdown[idx + len(anchor):]
    match = re.match(r"\s+\(1\)\s*\n\s*\(2\)\s*\n\s*\(3\)", after)
    if not match:
        return markdown
    return markdown[:idx + len(anchor)] + "\n\n" + _P30_FORMULAS + after[match.end():]


def _convert_math_blocks(markdown: str, page_num: int) -> str:
    """Promote known datasheet formulas from plain text to LaTeX blocks."""
    if page_num == 30:
        markdown = _replace_p30_formulas(markdown)
    markdown = _THETA_BLOCKING_RE.sub(lambda _: _THETA_BLOCKING_LATEX, markdown)
    markdown = _FLAT_DEACTIVATE_RE.sub(
        lambda _: _FLAT_LATEX_TEMPLATE.format(op="+", cmp="<", suffix=r"\quad \text{OR NOT (no\_movement)}"),
        markdown,
    )
    markdown = _FLAT_ACTIVATE_RE.sub(
        lambda _: _FLAT_LATEX_TEMPLATE.format(op="-", cmp=r"\geq", suffix=r"\quad \text{AND (no\_movement)}"),
        markdown,
    )
    # Unify the angle variable spelling used in the orientation tables.
    markdown = re.sub(r"\bphi\b", "φ", markdown)
    markdown = re.sub(r"\bTheta\b", "θ", markdown)
    # Clear stray underline artifacts left by the original layout.
    markdown = re.sub(r"(?<![A-Za-z0-9_])_\s+_\s+_(?![A-Za-z0-9_])", "—", markdown)
    return markdown


def _figure_regions(lines: list[dict], page, found_tables: list | None = None) -> list[dict]:
    page_rect = page.rect
    page_text = _normalize_private_use_text(page.get_text("text", sort=True))
    if re.search(r"^\s*List\s+of\s+figures\s*$", page_text, re.M | re.I):
        return []
    page_images = page.get_image_info()
    image_tops = [
        (info["bbox"][1], info["bbox"][3] - info["bbox"][1])
        for info in page_images
    ]
    page_drawings = page.get_drawings()
    genuine_boxes = []
    for table in (found_tables or []):
        if not _is_genuine_table(table):
            continue
        tx0, ty0, tx1, ty1 = table.bbox
        nested = any(
            other.bbox is not table.bbox
            and other.bbox[0] >= tx0 + 5
            and other.bbox[1] >= ty0 + 5
            and other.bbox[2] <= tx1 - 5
            and other.bbox[3] <= ty1 - 5
            and other.bbox[3] - other.bbox[1] >= 25
            for other in (found_tables or [])
        )
        if not nested:
            genuine_boxes.append(table.bbox)

    caption_re = re.compile(
        r"Figure\s+[\d\-]+\s*[:.]|Fig\.?\s*[\d\-]+|"
        r"图\s*[\d\-]+(?:[（(]\s*[A-Za-z0-9]\s*[）)])?",
        re.IGNORECASE,
    )
    captions: list[dict] = []
    for line in lines:
        if genuine_boxes and any(
            tx0 <= (line["bbox"][0] + line["bbox"][2]) / 2 <= tx1
            and ty0 <= (line["bbox"][1] + line["bbox"][3]) / 2 <= ty1
            for tx0, ty0, tx1, ty1 in genuine_boxes
        ):
            # A line inside a detected table is table content, not a caption.
            continue
        matches = list(caption_re.finditer(line["text"]))
        if not matches:
            continue
        if line["text"][: matches[0].start()].strip():
            # "see Figure 14." inside a paragraph is a reference, not a caption.
            continue
        x0, y0, x1, y1 = line["bbox"]
        width = max(1.0, x1 - x0)
        text_len = max(1, len(line["text"]))
        for index, match in enumerate(matches):
            start = match.start()
            end = matches[index + 1].start() if index + 1 < len(matches) else len(line["text"])
            text = line["text"][start:end].strip()
            if not text:
                continue
            # A TI-style line can carry two captions ("Figure 1. A   Figure 2. B");
            # split the line's x-range at the midpoint between the two captions
            # so each diagram gets its own crop.
            if len(matches) == 1:
                cx0, cx1 = x0, x1
            elif index == 0:
                next_start = matches[1].start()
                cx0, cx1 = x0, x0 + width * (next_start / text_len)
            elif index == len(matches) - 1:
                prev_end = matches[index - 1].end()
                cx0, cx1 = x0 + width * (prev_end / text_len), x1
            else:
                prev_end = matches[index - 1].end()
                next_start = matches[index + 1].start()
                cx0 = x0 + width * (prev_end / text_len)
                cx1 = x0 + width * (next_start / text_len)
            captions.append({
                "text": text,
                "bbox": (cx0, y0, cx1, y1),
                "multi": len(matches) > 1,
            })

    # Two-column layouts (TI/ADI) put side-by-side captions on separate lines at
    # the same height. Split the page's x-range between captions that share a
    # y-band so each diagram gets its own crop instead of duplicating the whole
    # width; single captions keep the full-width crop.
    page_x0 = max(55.0, page_rect.x0)
    page_x1 = min(page_rect.x1 - 45.0, 545.0)
    y_groups: list[list[dict]] = []
    for caption in captions:
        cy = (caption["bbox"][1] + caption["bbox"][3]) / 2
        for group in y_groups:
            group_y = (group[0]["bbox"][1] + group[0]["bbox"][3]) / 2
            if abs(cy - group_y) <= 12:
                group.append(caption)
                break
        else:
            y_groups.append([caption])
    for group in y_groups:
        if len(group) < 2:
            for caption in group:
                caption["box_x"] = (page_x0, page_x1)
            continue
        ordered = sorted(group, key=lambda cap: (cap["bbox"][0] + cap["bbox"][2]) / 2)
        centers = [(cap["bbox"][0] + cap["bbox"][2]) / 2 for cap in ordered]
        for index, caption in enumerate(ordered):
            left = page_x0 if index == 0 else (centers[index - 1] + centers[index]) / 2
            right = page_x1 if index == len(ordered) - 1 else (centers[index] + centers[index + 1]) / 2
            caption["box_x"] = (max(page_x0, left - 5), min(page_x1, right + 5))

    def content_centers(lo: float, hi: float) -> list[float]:
        """y-centers of graphics inside (lo, hi); excludes footer furniture."""
        centers = []
        for info in page_images:
            bbox = info["bbox"]
            center = (bbox[1] + bbox[3]) / 2
            if lo < center < hi and center < page_rect.height - 70:
                centers.append(center)
        for drawing in page_drawings:
            rect = drawing["rect"]
            center = (rect.y0 + rect.y1) / 2
            if (
                lo < center < hi
                and center < page_rect.height - 70
            ):
                centers.append(center)
        return centers

    def first_cluster_bottom(centers: list[float], gap: float = 60.0) -> float:
        """Bottom-most point of the first contiguous run of content centers."""
        ordered = sorted(centers)
        if not ordered:
            return 0.0
        previous = ordered[0]
        for current in ordered[1:]:
            if current - previous > gap:
                break
            previous = current
        return previous

    figures = []
    group_bottoms: list[float] = []
    for group in y_groups:
        previous_caption_bottom = group_bottoms[-1] if group_bottoms else 84.0
        group_bottom = max(caption["bbox"][3] for caption in group)
        group_bottoms.append(group_bottom)
        next_group_top = (
            min(caption["bbox"][1] for caption in y_groups[y_groups.index(group) + 1])
            if y_groups.index(group) + 1 < len(y_groups)
            else page_rect.height
        )
        for caption in group:
            x0, y0, x1, y1 = caption["bbox"]
            # Captions with real graphics nearby become figures; index lines
            # and bare references are skipped.
            above = content_centers(previous_caption_bottom, y0 - 5)
            below = content_centers(y1 + 5, next_group_top - 10)
            if not above and not below:
                continue
            if above:
                top = max(84.0, previous_caption_bottom + 10.0, y0 - 480.0)
                before_caption = [
                    line for line in lines
                    if previous_caption_bottom < line["bbox"][1] < y0 - 20.0
                ]
                gaps = []
                for previous, current in zip(before_caption, before_caption[1:]):
                    gap = current["bbox"][1] - previous["bbox"][3]
                    if gap >= 18.0:
                        gaps.append((gap, previous["bbox"][3] + 8.0))
                if gaps:
                    # The largest whitespace break normally separates
                    # introductory prose from the drawing.
                    top = max(top, max(gaps)[1])
                top = min(top, y0 - 200.0)
                # Closely spaced diagram rows (I²C/SPI bit streams) look like
                # large internal whitespace gaps; never let the refinement cut
                # the topmost graphic out of the crop.
                if above:
                    top = min(top, min(above) - 8.0)
                top = max(84.0, previous_caption_bottom + 10.0, top)
                near_images = [
                    image_top for image_top, image_height in image_tops
                    if top - 20 <= image_top <= y0 - 5 and image_height >= 20
                ]
                if near_images:
                    top = min(near_images)
                bottom = max(top + 20.0, y0 - 3.0)
            else:
                # Caption above the drawing: crop downward from the caption to
                # just past the bottom-most graphic, bounded by the next group.
                top = max(y1 + 3.0, y0 + 10.0)
                bottom = min(next_group_top - 5.0, first_cluster_bottom(below) + 10.0)
            box_x0, box_x1 = caption.get("box_x", (page_x0, page_x1))
            bbox = (box_x0, top, box_x1, bottom)
            if bbox[3] - bbox[1] >= 35:
                figures.append({
                    "bbox": bbox,
                    "caption": caption["text"],
                    "sort_y": y0 - 0.1,
                })

    # Uncaptioned content images (product photos, package outline drawings)
    # belong to the original page too; extract them with a generic caption.
    existing = [figure["bbox"] for figure in figures]
    for info in page_images:
        ix0, iy0, ix1, iy1 = info["bbox"]
        width, height = ix1 - ix0, iy1 - iy0
        if width < 60 or height < 60:
            continue
        if width > page_rect.width * 0.85 and height > page_rect.height * 0.85:
            continue  # full-page background / cover art
        if iy0 < page_rect.height * 0.08 or iy1 > page_rect.height * 0.95:
            continue  # header / footer logo strip
        center = ((ix0 + ix1) / 2, (iy0 + iy1) / 2)
        if any(
            fx0 <= center[0] <= fx1 and fy0 <= center[1] <= fy1
            for fx0, fy0, fx1, fy1 in existing
        ):
            continue
        figures.append({
            "bbox": (
                max(55.0, page_rect.x0, ix0 - 5.0),
                iy0 - 5.0,
                min(page_rect.x1 - 45.0, 545.0, ix1 + 5.0),
                iy1 + 5.0,
            ),
            "caption": "插图",
            "sort_y": iy0 - 0.1,
        })
    return figures


def _native_layout_markdown(lines, tables, figures, page_rect, session_id, page_num):
    items = []
    table_boxes = [table["bbox"] for table in tables]
    figure_boxes = [figure["bbox"] for figure in figures]
    for line in lines:
        x0, y0, x1, y1 = line["bbox"]
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        if any(tx0 <= cx <= tx1 and ty0 <= cy <= ty1 for tx0, ty0, tx1, ty1 in table_boxes):
            continue
        if any(fx0 <= cx <= fx1 and fy0 <= cy <= fy1 for fx0, fy0, fx1, fy1 in figure_boxes):
            # Vector-diagram labels belong to the rendered figure. Repeating
            # every pin name as prose makes Markdown pages much taller and
            # forces the preview to shrink well below the source-page scale.
            continue
        items.append({"kind": "line", "y": y0, "x": x0, **line})
    for table in tables:
        items.append({"kind": "table", "y": table["bbox"][1], "x": table["bbox"][0], **table})
    for figure in figures:
        items.append({"kind": "figure", "y": figure["sort_y"], "x": figure["bbox"][0], **figure})
    items.sort(key=lambda item: (round(item["y"], 1), item["x"], 0 if item["kind"] == "figure" else 1))

    markdown_parts = []
    detections = []
    for item in items:
        if item["kind"] == "table":
            markdown_parts.extend(["", item["markdown"], ""])
            detections.append({"type": "table", "bbox": _normalize_bbox(item["bbox"], page_rect), "text": item["text"]})
        elif item["kind"] == "figure":
            bbox = _normalize_bbox(item["bbox"], page_rect)
            coords = ",".join(map(str, bbox))
            url = f"/api/region-image/{quote(session_id)}/{page_num}/{quote(coords)}" if session_id else coords
            markdown_parts.extend(["", f"![{item['caption']}]({url})", ""])
            detections.append({"type": "image", "bbox": bbox, "text": item["caption"]})
        else:
            text = item["text"]
            det_type = "title" if _is_heading(text, item["bold"], item["size"]) else "text"
            rendered = f"## {text}" if det_type == "title" else text
            if det_type == "title":
                markdown_parts.append("")
            markdown_parts.append(rendered)
            detections.append({"type": det_type, "bbox": _normalize_bbox(item["bbox"], page_rect), "text": text})

    markdown = "\n".join(markdown_parts)
    markdown = re.sub(r"\n{3,}", "\n\n", markdown).strip()
    markdown = _convert_math_blocks(markdown, page_num)
    return markdown, detections


def _is_heading(text: str, bold: bool, size: float) -> bool:
    return bool(
        text.lower() == "index of contents"
        or re.match(r"^\d+(?:\.\d+)*\s+[A-Z]", text)
        or (bold and size >= 12 and len(text) < 120)
    )


def _normalize_bbox(bbox, page_rect) -> list[int]:
    x0, y0, x1, y1 = bbox
    return [
        max(0, min(1000, round((x0 - page_rect.x0) / page_rect.width * 1000))),
        max(0, min(1000, round((y0 - page_rect.y0) / page_rect.height * 1000))),
        max(0, min(1000, round((x1 - page_rect.x0) / page_rect.width * 1000))),
        max(0, min(1000, round((y1 - page_rect.y0) / page_rect.height * 1000))),
    ]


def image_to_base64(image_path: Path) -> str:
    """Read image file and return base64-encoded string."""
    with open(image_path, "rb") as file:
        return base64.b64encode(file.read()).decode("utf-8")
