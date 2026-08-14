"""Parse OCR raw output into structured detections and HTML."""
import re
from urllib.parse import quote
from typing import Optional


def parse_ocr_output(raw: str) -> list[dict]:
    """Parse <|det|>type [bbox]<|/det|>text into structured list."""
    if not raw:
        return []

    detections = []
    pattern = re.compile(r"<\|det\|>(\w+)\s*\[([^\]]*)\]<\|/det\|>(.+)")
    for line in raw.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        if _is_model_artifact(line):
            continue
        m = pattern.match(line)
        if m:
            det_type = m.group(1)
            bbox_str = m.group(2)
            text = m.group(3).strip()
            try:
                bbox = [int(x.strip()) for x in bbox_str.split(",") if x.strip()]
            except ValueError:
                # A truncated/looping model response must not abort the entire
                # scan merely because one grounding coordinate is malformed.
                bbox = []
            detections.append({"type": det_type, "bbox": bbox, "text": text})
        else:
            # Fallback: treat whole line as text
            if line and not line.startswith("<|") :
                detections.append({"type": "text", "bbox": [], "text": line})

    return detections


def native_text_to_markdown(text: str) -> str:
    """Turn a digital PDF text layer into readable, conservative Markdown."""
    lines = []
    for raw_line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped:
            if lines and lines[-1] != "":
                lines.append("")
            continue
        if stripped.lower() == "index of contents":
            lines.append(f"# {stripped}")
        elif re.match(r"^\d+\.\s+[A-Z][A-Z\s,&/()\-]+(?:\.{2,}\s*\d+)?$", stripped):
            lines.append(f"## {stripped}")
        else:
            lines.append(stripped)
    return "\n".join(lines).strip()


def native_text_to_detections(text: str) -> list[dict]:
    """Create editable result blocks from an embedded PDF text layer."""
    detections = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        is_heading = (
            line.lower() == "index of contents"
            or bool(re.match(r"^\d+\.\s+[A-Z][A-Z\s,&/()\-]+", line))
        )
        detections.append({"type": "title" if is_heading else "text", "bbox": [], "text": line})
    return detections


def _is_model_artifact(line: str) -> bool:
    """Return true for grounding/debug fragments that are not document text."""
    stripped = line.strip()
    if stripped.lower() in {"[non-text]", "[non text]", "[non_text]"}:
        return True
    if "<|det|>" in stripped and "<|/det|>" not in stripped:
        return True
    return bool(re.fullmatch(
        r"(?:text|title|image|figure|chart|diagram|table|formula|page_number)\s*\[[\d\s,.-]*\]",
        stripped,
        flags=re.IGNORECASE,
    ))


def is_repetitive_stream_artifact(markdown: str) -> bool:
    """Detect an unstable partial decode without rejecting final OCR text."""
    lines = [line.strip() for line in markdown.splitlines() if line.strip()]
    if len(lines) < 3:
        return False

    normalized = [
        re.sub(
            r"^(?:text|title|image|figure|chart|diagram|table|formula|page_number)\s*\[[^\]]*\]\s*",
            "",
            line,
            flags=re.IGNORECASE,
        ).strip()
        for line in lines
    ]
    normalized = [line for line in normalized if line]
    if len(normalized) < 3:
        return False

    counts = {line: normalized.count(line) for line in set(normalized)}
    most_repeated = max(counts.values(), default=0)
    return most_repeated >= 3 and most_repeated / len(normalized) >= 0.5


def reconstruct_structure(detections: list[dict]) -> list[dict]:
    """Group detections into logical blocks (headings, paragraphs, images, etc.)."""
    if not detections:
        return []

    blocks = []
    current_para = None

    for det in detections:
        det_type = det["type"]

        if det_type == "title":
            # Flush current paragraph
            if current_para:
                blocks.append(current_para)
                current_para = None
            # Determine heading level from bbox height (larger = higher level)
            bbox = det.get("bbox", [])
            level = 2  # default
            if len(bbox) >= 4:
                h = bbox[3] - bbox[1]
                if h > 60:
                    level = 1
            blocks.append({
                "block_type": "heading",
                "level": level,
                "text": det["text"],
                "bbox": bbox,
            })

        elif det_type == "image":
            if current_para:
                blocks.append(current_para)
                current_para = None
            blocks.append({
                "block_type": "image",
                "text": det["text"],
                "bbox": det.get("bbox", []),
            })

        elif det_type == "page_number":
            if current_para:
                blocks.append(current_para)
                current_para = None
            blocks.append({
                "block_type": "page_number",
                "text": det["text"],
                "bbox": det.get("bbox", []),
            })

        elif det_type == "table":
            if current_para:
                blocks.append(current_para)
                current_para = None
            blocks.append({
                "block_type": "table",
                "text": det["text"],
                "bbox": det.get("bbox", []),
            })

        else:  # text and others
            if current_para is None:
                current_para = {
                    "block_type": "paragraph",
                    "text": det["text"],
                    "bbox": det.get("bbox", []),
                }
            else:
                # Append to existing paragraph
                current_para["text"] += "\n" + det["text"]

    if current_para:
        blocks.append(current_para)

    return blocks


def generate_html(detections: list[dict], page_num: int = 1) -> str:
    """Convert detections to HTML for the right panel.

    Uses the raw detections list (not blocks) so that data-detection-index
    values match 1:1 with the detections array index.  This is critical for
    the translation feature which looks up DOM elements by detection index.
    """
    if not detections:
        return '<div class="ocr-page empty"><p class="muted">No content detected</p></div>'

    parts = [f'<div class="ocr-page" data-page="{page_num}">']

    for i, det in enumerate(detections):
        det_type = det.get("type", "text")
        text = _escape_html(det.get("text", ""))
        idx_attr = f'data-detection-index="{i}"'

        if det_type == "title":
            bbox = det.get("bbox", [])
            h = (bbox[3] - bbox[1]) if len(bbox) >= 4 else 0
            level = 1 if h > 60 else 2
            tag = f"h{min(level, 3)}"
            parts.append(
                f'<{tag} class="ocr-heading" contenteditable="true" '
                f'{idx_attr}>{text}</{tag}>'
            )

        elif det_type == "image":
            parts.append(
                f'<div class="ocr-image" {idx_attr}>'
                f'<span class="image-placeholder">🖼 图片区域</span>'
                f'</div>'
            )

        elif det_type == "table":
            parts.append(
                f'<div class="ocr-table" contenteditable="true" '
                f'{idx_attr}>{text}</div>'
            )

        elif det_type == "page_number":
            parts.append(
                f'<span class="ocr-page-number" {idx_attr}>{text}</span>'
            )

        else:  # text and others
            parts.append(
                f'<p class="ocr-text" contenteditable="true" '
                f'{idx_attr}>{text}</p>'
            )

    parts.append('</div>')
    return "\n".join(parts)


def _escape_html(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def raw_to_markdown(raw: str, session_id: str = "", page_num: int = 1) -> str:
    """Remove grounding metadata while preserving model-produced Markdown."""
    if not raw:
        return ""

    pattern = re.compile(r"<\|det\|>(\w+)\s*\[([^\]]*)\]<\|/det\|>")

    def replace_detection(match: re.Match) -> str:
        det_type = match.group(1).lower()
        coords = ",".join(x.strip() for x in match.group(2).split(","))
        if det_type in {"image", "figure", "chart", "diagram"} and session_id:
            url = f"/api/region-image/{quote(session_id)}/{page_num}/{quote(coords)}"
            return f"\n\n![图表或图片区域]({url})\n\n"
        if det_type == "title":
            return "\n\n## "
        if det_type in {"table", "formula"}:
            return "\n\n"
        return ""

    markdown = pattern.sub(replace_detection, raw)
    # Some pages make the model emit internal non-text placeholders or a
    # truncated grounding record. They are useful to the model but must never
    # appear in the document preview/export.
    markdown = re.sub(r"(?im)^\s*\[non[-_ ]?text\]\s*$", "", markdown)
    markdown = re.sub(r"(?im)^\s*<\|det\|>[^\n]*(?:<\|/det\|>)?\s*$", "", markdown)
    markdown = re.sub(
        r"(?im)^\s*(?:text|title|image|figure|chart|diagram|table|formula|page_number)\s*\[[\d\s,.-]*\]\s*$",
        "",
        markdown,
    )
    markdown = re.sub(r"<\|/?det\|>", "", markdown)
    markdown = re.sub(r"<\|/?(?:ref|grounding)\|>", "", markdown)
    markdown = markdown.replace("<｜end▁of▁sentence｜>", "")
    markdown = re.sub(r"^(#{1,6})\s+(#{1,6})\s+", r"\2 ", markdown, flags=re.MULTILINE)
    markdown = re.sub(r"\n{3,}", "\n\n", markdown)
    return markdown.strip()


def generate_det_html(det: dict, index: int) -> str:
    """Generate HTML for a single detection (for real-time line-by-line push)."""
    det_type = det.get("type", "text")
    text = _escape_html(det.get("text", ""))

    if det_type == "title":
        bbox = det.get("bbox", [])
        h = (bbox[3] - bbox[1]) if len(bbox) >= 4 else 0
        level = 1 if h > 60 else 2
        tag = f"h{min(level, 3)}"
        return f'<{tag} class="ocr-heading" contenteditable="true" data-detection-index="{index}">{text}</{tag}>'

    elif det_type == "image":
        return f'<div class="ocr-image" data-detection-index="{index}"><span class="image-placeholder">🖼 图片区域</span></div>'

    elif det_type == "table":
        return f'<div class="ocr-table" contenteditable="true" data-detection-index="{index}">{text}</div>'

    elif det_type == "page_number":
        return f'<span class="ocr-page-number" data-detection-index="{index}">{text}</span>'

    else:  # text and others
        return f'<p class="ocr-text" contenteditable="true" data-detection-index="{index}">{text}</p>'
