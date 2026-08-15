"""Region-level Hybrid OCR.

For a digital PDF page with a reliable text layer plus complex regions
(tables, figures), Hybrid keeps the body text native and runs the visual
model only on selected regions. Native regions are recovered verbatim;
region OCR failures degrade to the native region and become warnings
instead of failing the page.

First version scope: only low-quality TABLE regions trigger VLM OCR.
Figures stay image crops; formula OCR is a structured extension point
plugged into the same region pipeline.
"""
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

# Tables with native-extraction confidence below this get region-level VLM OCR.
TABLE_OCR_THRESHOLD = 0.70


@dataclass
class DocumentRegion:
    region_id: str
    type: str  # text | title | table | formula | figure | image
    bbox: tuple  # page coordinates (x0, y0, x1, y1)
    normalized_bbox: list  # 0..1000 normalized page bbox
    source: str  # native | vlm
    content: str
    confidence: float | None = None


def assess_native_table_quality(region: dict) -> float:
    """Heuristic 0..1 confidence for a native-extracted table region.

    Rewards consistent column counts across rows, low empty-cell ratio and
    enough structure. Below TABLE_OCR_THRESHOLD the region is a VLM candidate.
    """
    content = (region.get("content") or "").strip()
    if not content:
        return 0.0
    lines = [ln for ln in content.splitlines() if ln.strip()]
    if len(lines) < 2:
        return 0.0
    cell_rows = []
    for ln in lines:
        cells = [c.strip() for c in ln.split("|")]
        cells = [c for c in cells if c != ""]
        if cells:
            cell_rows.append(cells)
    if not cell_rows:
        return 0.0
    col_counts = [len(row) for row in cell_rows]
    counts = Counter(col_counts)
    modal_count, modal_hits = counts.most_common(1)[0]
    modal_ratio = modal_hits / len(col_counts)
    max_cols = max(col_counts)
    grid_cells = max_cols * len(cell_rows)
    empty_ratio = 1.0 - (sum(col_counts) / grid_cells) if grid_cells else 1.0
    inconsistent = 1 - modal_ratio
    score = modal_ratio * 0.5 + (1 - empty_ratio) * 0.3 + (1 - inconsistent) * 0.2
    return round(min(1.0, max(0.0, score)), 3)


def map_region_bbox_to_page(local_bbox, region_normalized_bbox) -> list[int]:
    """Map a 0..1000 normalized bbox inside a cropped region back to page coords.

    region_normalized_bbox is the page bbox of the crop in 0..1000 space.
    Returns the page-normalized bbox clamped to 0..1000. Pure function; a
    malformed input returns [].
    """
    if len(local_bbox) != 4 or len(region_normalized_bbox) != 4:
        return []
    try:
        lx1, ly1, lx2, ly2 = [int(float(v)) for v in local_bbox]
        rx1, ry1, rx2, ry2 = [int(float(v)) for v in region_normalized_bbox]
    except (TypeError, ValueError):
        return []

    def clamp(v):
        return max(0, min(1000, v))

    w = rx2 - rx1
    h = ry2 - ry1
    return [
        clamp(rx1 + round(lx1 / 1000 * w)),
        clamp(ry1 + round(ly1 / 1000 * h)),
        clamp(rx1 + round(lx2 / 1000 * w)),
        clamp(ry1 + round(ly2 / 1000 * h)),
    ]


def crop_page_region(
    image_path: Path | str,
    normalized_bbox: list[int],
    output_path: Path | str,
    padding: int = 8,
) -> Path:
    """Crop a region from a rendered page image.

    Coordinates are in Unlimited-OCR normalized 0..1000 space; they are
    scaled to the actual image size and clamped to the image bounds.
    """
    from PIL import Image

    x1, y1, x2, y2 = normalized_bbox
    with Image.open(str(image_path)) as image:
        width, height = image.size
        px1 = max(0, int(x1 / 1000 * width) - padding)
        py1 = max(0, int(y1 / 1000 * height) - padding)
        px2 = min(width, int(x2 / 1000 * width) + padding)
        py2 = min(height, int(y2 / 1000 * height) + padding)
        if px2 - px1 < 1 or py2 - py1 < 1:
            raise ValueError(f"region too small to crop: {normalized_bbox}")
        image.crop((px1, py1, px2, py2)).save(str(output_path), "PNG")
    return Path(output_path)


def ocr_region(image_path, region: dict, ocr_engine, max_length: int, crop_dir) -> tuple[str, list[dict]]:
    """OCR one region by cropping it and running the visual model.

    Returns (text, detections) with detection bboxes mapped back to page
    coordinates. Raises on OCR failure so callers can degrade to native.
    """
    from ocr_parser import parse_ocr_output

    normalized_bbox = region["normalized_bbox"]
    crop_dir = Path(crop_dir)
    crop_dir.mkdir(parents=True, exist_ok=True)
    crop_path = crop_dir / f"{region['region_id']}.png"
    crop_page_region(image_path, normalized_bbox, crop_path)
    raw = ocr_engine.ocr_page(str(crop_path), max_length)
    local_dets = parse_ocr_output(raw) or []
    detections = []
    for det in local_dets:
        det = dict(det)
        if len(det.get("bbox", [])) == 4:
            det["bbox"] = map_region_bbox_to_page(det["bbox"], normalized_bbox)
        det["source"] = "vlm"
        detections.append(det)
    text = "\n".join(d["text"] for d in detections)
    return text, detections


def merge_hybrid_regions(native_regions: list[dict], replacement_regions: list[dict]) -> list[dict]:
    """Return regions with replacements applied, sorted by visual order.

    A replacement region overrides the native region with the same region_id;
    everything else is kept. The result is sorted by (y1, x1).
    """
    by_id = {r["region_id"]: r for r in replacement_regions if r.get("region_id")}
    merged = []
    for region in native_regions:
        if region.get("region_id") in by_id:
            merged.append(by_id[region["region_id"]])
        else:
            merged.append(region)
    return sorted(merged, key=lambda r: (r.get("bbox", (0, 0, 0, 0))[1], r.get("bbox", (0, 0, 0, 0))[0]))


def build_hybrid_result(session, page_num: int, native_page: dict, engine, max_length: int) -> dict:
    """Run region-level hybrid OCR for one page and build its page result.

    Only low-quality table regions invoke the visual model. Region failures
    keep the native region and are collected in warnings. Returns the unified
    page result dict (detections / markdown / html / blocks / source / warnings).
    """
    from ocr_parser import generate_html, reconstruct_structure
    from pdf_converter import _convert_math_blocks

    regions = native_page.get("regions") or []
    image_path = str(session.page_images[page_num - 1])
    crop_dir = Path(session.upload_dir) / "hybrid"
    warnings = []
    replacements = []

    for region in regions:
        if region.get("type") != "table":
            continue  # figures stay image crops; formula OCR is an extension point
        quality = assess_native_table_quality(region)
        if quality >= TABLE_OCR_THRESHOLD:
            continue
        try:
            text, detections = ocr_region(image_path, region, engine, max_length, crop_dir)
            if not detections or not text.strip():
                warnings.append({
                    "type": "region_ocr_empty",
                    "region_id": region["region_id"],
                    "message": "VLM returned no detections for region",
                })
                continue
            replacement = dict(region)
            replacement["source"] = "vlm"
            replacement["content"] = text
            replacement["detections"] = detections
            replacement["confidence"] = quality
            replacements.append(replacement)
        except Exception as exc:
            warnings.append({
                "type": "region_ocr_failed",
                "region_id": region["region_id"],
                "message": str(exc),
            })
            continue

    merged = merge_hybrid_regions(regions, replacements)
    detections = []
    markdown_parts = []
    for region in merged:
        region_type = region.get("type", "text")
        content = (region.get("content") or "").strip()
        nb = region.get("normalized_bbox") or []
        source = region.get("source", "native")
        if region_type == "table":
            detections.append({"type": "table", "bbox": nb, "text": content, "source": source})
            if content:
                markdown_parts.extend(["", content, ""])
        elif region_type == "figure":
            coords = ",".join(map(str, nb)) if len(nb) == 4 else ""
            url = f"/api/region-image/{session.session_id}/{page_num}/{coords}" if coords else ""
            detections.append({"type": "image", "bbox": nb, "text": content, "source": source})
            if url:
                markdown_parts.extend(["", f"![{content}]({url})", ""])
        else:
            det_type = region_type if region_type in ("title", "text") else "text"
            detections.append({"type": det_type, "bbox": nb, "text": content, "source": source})
            markdown_parts.append(f"## {content}" if det_type == "title" and content else content)

    markdown = re.sub(r"\n{3,}", "\n\n", "\n".join(markdown_parts)).strip()
    markdown = _convert_math_blocks(markdown, page_num)
    html = generate_html(detections, page_num)
    blocks = reconstruct_structure(detections)
    return {
        "detections": detections,
        "markdown": markdown,
        "html": html,
        "blocks": blocks,
        "raw": markdown,
        "source": "hybrid",
        "truncated": False,
        "warnings": warnings,
    }
