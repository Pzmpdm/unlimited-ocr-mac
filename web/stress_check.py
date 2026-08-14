"""Page-by-page stress checker: original PDF vs generated Markdown.

Usage:
    python stress_check.py [--verbose] datasheet1.pdf [datasheet2.pdf ...]

For every page it compares the PDF text layer, table cells, figure captions and
math-like lines against the Markdown produced by pdf_converter, and reports any
page where content is lost or mangled.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pymupdf as fitz

from pdf_converter import (
    extract_native_pages,
    _normalize_private_use_text,
    _page_lines,
    _figure_regions,
)


def norm(value: str) -> str:
    """Whitespace-free normalization used for fuzzy content comparison."""
    value = re.sub(r"\s+", "", value)
    value = re.sub(r"[\u200b\u00ad]", "", value)
    return value.lower()


def significant(value: str) -> str:
    """Keep only meaningful characters so table pipes/whitespace don't matter."""
    return re.sub(
        r"[^0-9a-zA-Z\u4e00-\u9fffθφσΩωαβγδπλμ°±×≤≥·µΔ]",
        "",
        norm(value),
    )


def digit_tokens(text: str) -> set[str]:
    """Unique numeric tokens (numbers with optional sign/unit markers)."""
    tokens = set()
    for token in re.findall(r"[0-9][0-9.,:/\-]*", text):
        token = token.strip(".,:/-")
        if len(token) >= 2:
            tokens.add(token)
    return tokens


def math_like(line: str) -> bool:
    line = line.strip()
    if len(line) < 12:
        return False
    return bool(
        "=" in line
        or re.search(r"√|±|×|·|≤|≥|°", line)
        or re.search(r"\b(sin|cos|tan|log|exp|sqrt)\b", line, re.I)
        or re.search(r"[θφσΩωαβγδπλμ]", line)
    )


def markdown_table_blocks(markdown: str) -> list[str]:
    blocks: list[str] = []
    current: list[str] = []
    for line in markdown.splitlines():
        if line.startswith("| "):
            current.append(line)
        elif current:
            blocks.append("\n".join(current))
            current = []
    if current:
        blocks.append("\n".join(current))
    return blocks


def is_real_table(table) -> bool:
    """A genuine table has readable content in several rows and columns."""
    if table.bbox[3] - table.bbox[1] < 25 or table.bbox[2] - table.bbox[0] < 180:
        return False
    if len(table.rows) < 2:
        return False
    rows = table.extract()
    populated = [
        [cell for cell in row if cell and cell.strip()]
        for row in rows
    ]
    multi_col_rows = [row for row in populated if len(row) >= 2]
    return len(multi_col_rows) >= 2 and sum(len(row) for row in populated) >= 6


def check_pdf(pdf_path: Path, verbose: bool = False) -> list[dict]:
    issues: list[dict] = []
    doc = fitz.open(pdf_path)
    pages = extract_native_pages(str(pdf_path))
    try:
        for page_num, (page, native) in enumerate(zip(doc, pages), 1):
            raw_text = _normalize_private_use_text(page.get_text("text", sort=True))
            markdown = native["markdown"]
            md_norm = norm(markdown)
            md_sig = significant(markdown)
            page_issues: list[str] = []
            lines = _page_lines(page)
            try:
                found_tables = page.find_tables().tables
            except Exception:
                found_tables = []
            figures = _figure_regions(lines, page, found_tables)
            figure_boxes = [figure["bbox"] for figure in figures]
            has_footnotes = bool(
                re.search(r"(?m)^\s*\d+\s+[A-Za-z(]", raw_text)
            )

            def in_figure(y_center: float) -> bool:
                if y_center is None:
                    return False
                return any(
                    fy0 <= y_center <= fy1
                    for fx0, fy0, fx1, fy1 in figure_boxes
                )

            raw_lines = raw_text.splitlines()
            prose_lines = [
                line for line in raw_lines
                if not in_figure(_line_center(line, lines))
            ]

            # ── 1. Text coverage ──────────────────────────────────────
            raw_norm = norm("\n".join(prose_lines))
            if raw_norm:
                covered = sum(1 for ch in raw_norm if ch in md_norm)
                coverage = covered / len(raw_norm)
                if coverage < 0.82:
                    page_issues.append(f"text coverage {coverage:.0%} < 82%")

            # ── 2. Tables ────────────────────────────────────────────
            def overlaps_figure(bbox) -> bool:
                cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
                return any(
                    fx0 <= cx <= fx1 and fy0 <= cy <= fy1
                    for fx0, fy0, fx1, fy1 in figure_boxes
                )

            genuine = [t for t in found_tables if is_real_table(t)]
            real_tables = [
                t for t in genuine
                if not any(
                    other.bbox is not t.bbox
                    and other.bbox[0] >= t.bbox[0] + 5
                    and other.bbox[1] >= t.bbox[1] + 5
                    and other.bbox[2] <= t.bbox[2] - 5
                    and other.bbox[3] <= t.bbox[3] - 5
                    and other.bbox[3] - other.bbox[1] >= 25
                    for other in genuine
                )
            ]
            real_tables = [t for t in real_tables if not overlaps_figure(t.bbox)]
            md_tables = markdown_table_blocks(markdown)
            if real_tables and not md_tables:
                page_issues.append(f"{len(real_tables)} table(s) lost")
            if real_tables:
                expected_digits: set[str] = set()
                for table in real_tables:
                    for row in table.extract():
                        for cell in row:
                            if cell:
                                expected_digits |= digit_tokens(cell)
                md_digits = digit_tokens(re.sub(r"\^\d+", "", markdown))
                missing_digits = sorted(
                    token for token in expected_digits
                    if token not in md_digits and len(token) >= 3
                )
                if has_footnotes:
                    missing_digits = [
                        token for token in missing_digits
                        if not (
                            token[-1].isdigit()
                            and token[:-1] in md_digits
                        )
                    ]
                if missing_digits:
                    page_issues.append(
                        f"table digits missing: {missing_digits[:12]}{'…' if len(missing_digits) > 12 else ''}"
                    )

            # ── 3. Figures ───────────────────────────────────────────
            if re.search(r"(?m)^\s*List\s+of\s+figures\s*$", raw_text, re.I):
                captions = []
            else:
                captions = re.findall(
                    r"(?m)^\s*(?:Figure\s+[\d\-]+\s*[:.]|Fig\.?\s*[\d\-]+|"
                    r"图\s*[\d\-]+(?:[（(]\s*[A-Za-z0-9]\s*[）)])?)",
                    raw_text,
                    re.IGNORECASE,
                )
            md_images = markdown.count("![")
            if captions and md_images < len(captions):
                page_issues.append(
                    f"figures: {len(captions)} caption(s) but {md_images} image(s) in md"
                )
            content_images = [
                info for info in page.get_image_info()
                if _is_content_image(info, page.rect, figure_boxes)
            ]
            if content_images and md_images == 0 and not captions:
                page_issues.append(
                    f"{len(content_images)} content image(s) without caption not extracted"
                )

            # ── 4. Formulas / math lines ─────────────────────────────
            lost_math = []
            for line in prose_lines:
                if not math_like(line):
                    continue
                sig = significant(line)
                tokens = re.findall(r"[0-9a-zA-Z\u4e00-\u9fff]+", sig)
                if len(tokens) < 6:
                    continue
                missing = [token for token in tokens if token not in md_sig]
                if len(missing) / len(tokens) > 0.4:
                    lost_math.append(line.strip()[:60])
            if lost_math:
                page_issues.append(f"math lines lost: {lost_math[:4]}{'…' if len(lost_math) > 4 else ''}")

            if page_issues:
                issues.append({"page": page_num, "problems": page_issues})
                if verbose:
                    print(f"  P{page_num}: {'; '.join(page_issues)}")
    finally:
        doc.close()
    return issues


def _line_center(line: str, page_lines: list[dict]) -> float | None:
    """Best-effort y-center for a raw text line based on its first tokens."""
    probe = norm(line)[:20]
    for entry in page_lines:
        if norm(entry["text"]).startswith(probe):
            return (entry["bbox"][1] + entry["bbox"][3]) / 2
    return None


def _is_content_image(info: dict, page_rect, figure_boxes) -> bool:
    """Content figures are substantial, centered, and outside existing crops."""
    x0, y0, x1, y1 = info["bbox"]
    width, height = x1 - x0, y1 - y0
    if width < 60 or height < 60:
        return False
    page_w, page_h = page_rect.width, page_rect.height
    if width > page_w * 0.85 and height > page_h * 0.85:
        return False  # full-page background / cover art
    if y0 < page_h * 0.08 or y1 > page_h * 0.95:
        return False  # header / footer logo strip
    center_y = (y0 + y1) / 2
    if any(fy0 <= center_y <= fy1 for fx0, fy0, fx1, fy1 in figure_boxes):
        return False
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdfs", nargs="+", type=Path)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    failed = 0
    for pdf in args.pdfs:
        print(f"\n=== {pdf.name} ===")
        try:
            issues = check_pdf(pdf, verbose=args.verbose)
        except Exception as exc:  # noqa: BLE001
            print(f"  ERROR: {exc}")
            failed += 1
            continue
        pages = len(issues)
        total_problems = sum(len(issue["problems"]) for issue in issues)
        print(f"  pages with issues: {pages}, total problems: {total_problems}")
        if pages:
            failed += 1
            for issue in issues[:20]:
                print(f"  P{issue['page']}: {'; '.join(issue['problems'])}")
    print(f"\n{'ALL PASS' if failed == 0 else f'{failed} DATASHEET(S) HAVE ISSUES'}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
