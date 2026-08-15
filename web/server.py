"""Unlimited-OCR Web: FastAPI server with SSE streaming."""
from __future__ import annotations
import asyncio
import json
import re
import os
import shutil
import time
import uuid
import threading
import zipfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import pymupdf as fitz

# Load .env file before importing config
_dotenv = Path(__file__).parent / ".env"
if _dotenv.exists():
    for line in _dotenv.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())

from fastapi import FastAPI, UploadFile, File, Form, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

if os.environ.get("OCR_BACKEND", "mlx").lower() == "pytorch":
    import ocr_engine
else:
    import ocr_engine_mlx as ocr_engine
import pdf_converter
import ocr_parser
import translator
import docx_exporter
from urllib.parse import unquote
from config import PUBLIC_DIR, UPLOAD_DIR, PORT, HOST, DEFAULT_MAX_LENGTH


# ── Session store ──────────────────────────────────────────────

class SessionData:
    def __init__(self, session_id: str, upload_dir: Path, source_name: str,
                 total_pages: int, page_images: list[Path],
                 native_page_texts: Optional[list[str]] = None,
                 native_pages: Optional[list[dict]] = None):
        self.session_id = session_id
        self.upload_dir = upload_dir
        self.source_name = source_name
        self.total_pages = total_pages
        self.page_images = page_images
        self.native_page_texts = native_page_texts or [""] * total_pages
        self.native_pages = native_pages or [{} for _ in range(total_pages)]
        self.page_results: dict[int, dict] = {}  # page_num → {detections, html, raw, blocks}
        self.page_status: dict[int, dict] = {  # page_num → {state, error, attempts}
            page: {"state": "pending", "error": None, "attempts": 0}
            for page in range(1, total_pages + 1)
        }
        self.page_translations: dict[int, list] = {}
        self.created_at = time.time()
        self.processing = False
        self.processed_pages = 0
        self.processing_error: Optional[str] = None


sessions: dict[str, SessionData] = {}


class ScanControl:
    def __init__(self):
        self.paused = threading.Event()
        self.cancelled = threading.Event()


scan_controls: dict[str, ScanControl] = {}


# ── App lifespan ───────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Eagerly load model in background
    print("[server] Loading OCR model in background...")
    asyncio.create_task(asyncio.to_thread(ocr_engine.ensure_model))
    yield
    # Cleanup
    for sid, s in list(sessions.items()):
        if s.upload_dir.exists():
            shutil.rmtree(s.upload_dir, ignore_errors=True)


app = FastAPI(title="Unlimited-OCR Web", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=str(PUBLIC_DIR)), name="static")


# ── Routes ─────────────────────────────────────────────────────

@app.get("/")
async def index():
    return FileResponse(PUBLIC_DIR / "index.html")


@app.get("/api/health")
async def health():
    return {"ok": True, **ocr_engine.get_status()}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    """Upload a PDF or image file. Returns session metadata."""
    session_id = uuid.uuid4().hex[:8]
    session_dir = UPLOAD_DIR / session_id
    session_dir.mkdir(exist_ok=True)

    # Save uploaded file
    src_path = session_dir / file.filename
    with open(src_path, "wb") as f:
        content = await file.read()
        f.write(content)

    source_name = file.filename
    ext = Path(file.filename).suffix.lower()

    if ext == ".pdf":
        # PDFs with many pages take a while to render and parse. Return
        # immediately and let the front-end poll /api/upload-progress.
        total_pages = pdf_converter.page_count(str(src_path))
        session = SessionData(session_id, session_dir, source_name, total_pages, [], [], [])
        session.processing = True
        sessions[session_id] = session
        asyncio.create_task(asyncio.to_thread(_process_pdf_sync, session, str(src_path)))
    elif ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
        total_pages = 1
        session = SessionData(session_id, session_dir, source_name, total_pages, [src_path], [], [])
        sessions[session_id] = session
    else:
        shutil.rmtree(session_dir, ignore_errors=True)
        return JSONResponse({"error": f"Unsupported file type: {ext}"}, status_code=400)

    return {
        "session_id": session_id,
        "total_pages": total_pages,
        "source_name": source_name,
        "processing": session.processing,
    }


def _process_pdf_sync(session: SessionData, src_path: str) -> None:
    """Render pages and extract native layout, updating progress per page."""
    try:
        doc = fitz.open(src_path)
        try:
            images: list[Path] = []
            native_pages: list[dict] = []
            for page_num in range(1, len(doc) + 1):
                out_path = session.upload_dir / f"page_{page_num:04d}.png"
                pdf_converter.render_page_image(doc, page_num - 1, out_path)
                images.append(out_path)
                native_pages.append(
                    pdf_converter.extract_native_page(doc, page_num, session.session_id)
                )
                session.processed_pages = page_num
            session.page_images = images
            session.native_pages = native_pages
            session.native_page_texts = [page.get("text", "") for page in native_pages]
            session.total_pages = len(images)
            session.processing_error = None
        finally:
            doc.close()
    except Exception as exc:  # noqa: BLE001
        session.processing_error = str(exc)
    finally:
        session.processing = False


@app.get("/api/upload-progress/{session_id}")
async def upload_progress(session_id: str):
    """Report how many pages have been rendered/parsed for an upload."""
    session = sessions.get(session_id)
    if not session:
        return JSONResponse({"error": "Session not found"}, status_code=404)
    return {
        "processing": session.processing,
        "processed_pages": session.processed_pages,
        "total_pages": session.total_pages,
        "error": session.processing_error,
    }


# ── Page processing (shared by /api/scan and /api/scan-page) ─────

def _resolve_engine():
    """Return the module-level OCR engine (injectable for tests)."""
    return ocr_engine


def _route_page(native_page: dict, native_chars: int, force_mode: Optional[str] = None) -> str:
    """Choose native / hybrid / vlm for a page.

    Issue 4 replaces this heuristic with page_router.choose_page_route; until
    then it keeps the historical native_chars >= 120 behaviour so the page
    status / retry work can land independently.
    """
    if force_mode in ("native", "hybrid", "vlm"):
        if force_mode == "hybrid":
            # Hybrid lands with Issue 5; until then it behaves like native.
            return "native"
        return force_mode
    return "native" if native_chars >= 120 else "vlm"


async def _stream_native_page(session, page_num, control, native_page, native_text):
    """Stream events for a page rendered entirely from the PDF text layer."""
    markdown = native_page.get("markdown") or ocr_parser.native_text_to_markdown(native_text)
    detections = native_page.get("detections") or ocr_parser.native_text_to_detections(native_text)
    for det in detections:
        det.setdefault("source", "native")
    blocks = ocr_parser.reconstruct_structure(detections)
    html = ocr_parser.generate_html(detections, page_num)
    chunk_size = 320
    for end in range(chunk_size, len(markdown) + chunk_size, chunk_size):
        if control is not None:
            while control.paused.is_set() and not control.cancelled.is_set():
                await asyncio.sleep(0.08)
            if control.cancelled.is_set():
                return
        partial = markdown[:min(end, len(markdown))]
        yield {"event": "token", "data": {
            "page_num": page_num, "text": partial, "markdown": partial,
            "tokens": max(1, len(partial) // 4),
            "done": end >= len(markdown), "source": "native", "truncated": False,
        }}
        await asyncio.sleep(0.025)

    session.page_results[page_num] = {
        "detections": detections, "html": html, "raw": native_text,
        "markdown": markdown, "blocks": blocks,
        "truncated": False, "source": "native",
    }
    for i, det in enumerate(detections):
        yield {"event": "det_result", "data": {
            "page_num": page_num, "det_index": i,
            "detection": det, "html": ocr_parser.generate_det_html(det, i),
            "total_detections": len(detections),
        }}
    yield {"event": "page_done", "data": {
        "page_num": page_num, "html": html, "markdown": markdown,
        "source": "native", "truncated": False,
    }}
    yield {"event": "page_image", "data": {
        "page_num": page_num,
        "image_url": f"/api/page-image/{session.session_id}/{page_num}",
    }}


async def _stream_hybrid_page(session, page_num, control, max_length, engine,
                              native_page, native_text, native_chars):
    """Stream events for a page using region-level Hybrid OCR (Issue 5).

    Until hybrid_ocr lands, degrade to the native text layer so the route is
    safe to select and retry behaves predictably.
    """
    async for ev in _stream_native_page(session, page_num, control, native_page, native_text):
        yield ev


async def _stream_vlm_page(session, page_num, control, max_length, engine,
                           native_text, native_chars):
    """Stream events for a page recognized by the visual OCR model."""
    img_path = str(session.page_images[page_num - 1])
    session_id = session.session_id
    page_truncated = False
    if hasattr(engine, "ocr_page_stream"):
        stream = engine.ocr_page_stream(img_path, max_length)
        raw = ""
        stable_markdown = ""
        while True:
            if control is not None:
                while control.paused.is_set() and not control.cancelled.is_set():
                    await asyncio.sleep(0.08)
                if control.cancelled.is_set():
                    stream.close()
                    return
            done, item = await asyncio.to_thread(_next_stream_item, stream)
            if done:
                break
            raw = item["text"]
            if item.get("done"):
                page_truncated = bool(item.get("truncated", False))
            live_markdown = ocr_parser.raw_to_markdown(raw, session_id, page_num)
            # The converted tokenizer can briefly decode the first few snapshots
            # as one repeated phrase, then replace the whole prefix. Never
            # animate that visibly unstable draft.
            if item.get("done") or not ocr_parser.is_repetitive_stream_artifact(live_markdown):
                stable_markdown = live_markdown
            yield {"event": "token", "data": {
                "page_num": page_num, "text": raw, "markdown": stable_markdown,
                "tokens": item.get("tokens", 0),
                "done": item.get("done", False),
                "truncated": item.get("truncated", False),
                "stats": item.get("stats"),
            }}
    else:
        raw = await asyncio.to_thread(engine.ocr_page, img_path, max_length)

    # If a sparse digital page makes the visual model terminate implausibly
    # early, retain the complete native text instead.
    generated_markdown = ocr_parser.raw_to_markdown(raw, session_id, page_num)
    if native_chars >= 120 and len(re.sub(r"\s+", "", generated_markdown)) < native_chars * 0.35:
        raw = native_text
        markdown = ocr_parser.native_text_to_markdown(native_text)
        detections = ocr_parser.native_text_to_detections(native_text)
        source = "vlm"  # page ran via VLM; content came from the native layer
    else:
        markdown = generated_markdown
        detections = ocr_parser.parse_ocr_output(raw)
        source = "vlm"
    for det in detections:
        det.setdefault("source", source)

    yield {"event": "page_progress", "data": {"page_num": page_num, "status": "parsing"}}

    blocks = ocr_parser.reconstruct_structure(detections)
    html = ocr_parser.generate_html(detections, page_num)
    session.page_results[page_num] = {
        "detections": detections, "html": html, "raw": raw,
        "markdown": markdown, "blocks": blocks,
        "truncated": page_truncated, "source": source,
    }
    for i, det in enumerate(detections):
        yield {"event": "det_result", "data": {
            "page_num": page_num, "det_index": i,
            "detection": det, "html": ocr_parser.generate_det_html(det, i),
            "total_detections": len(detections),
        }}
    yield {"event": "page_done", "data": {
        "page_num": page_num, "html": html, "markdown": markdown,
        "truncated": page_truncated, "source": source,
    }}
    yield {"event": "page_image", "data": {
        "page_num": page_num, "image_url": f"/api/page-image/{session_id}/{page_num}",
    }}


async def process_page(
    session: SessionData,
    page_num: int,
    control: Optional[ScanControl] = None,
    max_length: int = DEFAULT_MAX_LENGTH,
    force_mode: Optional[str] = None,
    engine=None,
):
    """Process a single page end-to-end, yielding internal events.

    Events are {"event": str, "data": dict} and are shared by /api/scan and
    /api/scan-page so retries use the exact same OCR flow. Updates
    session.page_status and session.page_results. Page-level failures never
    raise: they are recorded as state=failed and surfaced as an "error" event
    so a long scan keeps going.
    """
    engine = engine or _resolve_engine()
    status = session.page_status.setdefault(page_num, {"state": "pending", "error": None, "attempts": 0})
    status["attempts"] = int(status.get("attempts", 0)) + 1
    status["state"] = "processing"
    status["error"] = None
    print(f"[ocr] page={page_num} attempt={status['attempts']}")
    yield {"event": "page_start", "data": {
        "page_num": page_num, "total_pages": session.total_pages, "attempt": status["attempts"],
    }}
    yield {"event": "page_progress", "data": {"page_num": page_num, "status": "scanning"}}

    try:
        native_text = session.native_page_texts[page_num - 1] if page_num <= len(session.native_page_texts) else ""
        native_page = session.native_pages[page_num - 1] if page_num <= len(session.native_pages) else {}
        native_chars = len(re.sub(r"\s+", "", native_text))
        route = _route_page(native_page, native_chars, force_mode)
        print(f"[router] page={page_num} route={route}")
        if route == "native":
            async for ev in _stream_native_page(session, page_num, control, native_page, native_text):
                yield ev
        elif route == "hybrid":
            async for ev in _stream_hybrid_page(session, page_num, control, max_length, engine,
                                                native_page, native_text, native_chars):
                yield ev
        else:
            async for ev in _stream_vlm_page(session, page_num, control, max_length, engine,
                                             native_text, native_chars):
                yield ev

        if control is not None and control.cancelled.is_set():
            status["state"] = "cancelled"
            return
        result = session.page_results.get(page_num) or {}
        status["state"] = "warning" if result.get("truncated") else "done"
        result["status"] = status["state"]
    except Exception as exc:
        status["state"] = "failed"
        status["error"] = str(exc)
        session.page_results[page_num] = {
            "status": "failed", "error": str(exc),
            "detections": [], "html": "", "markdown": "", "blocks": [], "raw": "",
            "truncated": False,
        }
        print(f"[ocr] page={page_num} failed error={exc}")
        yield {"event": "error", "data": {"page_num": page_num, "message": str(exc)}}


@app.post("/api/scan")
async def scan(request: Request):
    """SSE endpoint: stream generated tokens, then structured page results."""
    body = await request.json()
    session_id = body.get("session_id")
    max_length = int(body.get("max_length", DEFAULT_MAX_LENGTH))

    session = sessions.get(session_id)
    if not session:
        return JSONResponse({"error": "Session not found"}, status_code=404)
    while session.processing:
        await asyncio.sleep(0.1)
    if session.processing_error:
        return JSONResponse({"error": f"读取失败: {session.processing_error}"}, status_code=500)
    control = ScanControl()
    scan_controls[session_id] = control

    async def event_generator():
        stopped = False
        try:
            for page_num in range(1, session.total_pages + 1):
                if control.cancelled.is_set():
                    stopped = True
                    break
                async for ev in process_page(session, page_num, control=control, max_length=max_length):
                    yield _sse(ev["event"], ev["data"])
            states = [s.get("state") for s in session.page_status.values()]
            yield _sse("scan_stopped" if stopped else "scan_complete", {
                "session_id": session_id,
                "total_pages": session.total_pages,
                "done_pages": sum(1 for s in states if s in ("done", "warning")),
                "failed_pages": sum(1 for s in states if s == "failed"),
            })
        finally:
            scan_controls.pop(session_id, None)

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/api/scan-page/{session_id}/{page_num}")
async def scan_page(session_id: str, page_num: int, request: Request):
    """Retry a single page, streaming the same SSE events as /api/scan.

    Request body: {"max_length": 8192, "force_mode": "native"|"hybrid"|"vlm"|null}
    """
    session = sessions.get(session_id)
    if not session:
        return JSONResponse({"error": "Session not found"}, status_code=404)
    if page_num < 1 or page_num > session.total_pages:
        return JSONResponse({"error": "Page out of range"}, status_code=404)
    body = await request.json()
    max_length = int(body.get("max_length", DEFAULT_MAX_LENGTH))
    force_mode = body.get("force_mode") or None
    if force_mode not in (None, "native", "hybrid", "vlm"):
        return JSONResponse({"error": f"Invalid force_mode: {force_mode}"}, status_code=400)

    async def event_generator():
        async for ev in process_page(session, page_num, max_length=max_length, force_mode=force_mode):
            yield _sse(ev["event"], ev["data"])

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.post("/api/scan-control")
async def scan_control(request: Request):
    body = await request.json()
    control = scan_controls.get(body.get("session_id"))
    if not control:
        return {"ok": False, "state": "idle"}
    action = body.get("action")
    if action == "pause":
        control.paused.set()
        return {"ok": True, "state": "paused"}
    if action == "resume":
        control.paused.clear()
        return {"ok": True, "state": "running"}
    if action == "stop":
        control.cancelled.set()
        control.paused.clear()
        return {"ok": True, "state": "stopping"}
    return JSONResponse({"error": "Unknown action"}, status_code=400)


def _sse(event: str, data: dict) -> str:
    """Format a single SSE event for immediate flush."""
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def _next_stream_item(iterator):
    """Bridge a blocking Python generator into asyncio without leaking StopIteration."""
    try:
        return False, next(iterator)
    except StopIteration:
        return True, None


@app.get("/api/page-image/{session_id}/{page_num}")
async def page_image(session_id: str, page_num: int):
    session = sessions.get(session_id)
    if not session or page_num < 1 or page_num > session.total_pages:
        return JSONResponse({"error": "Not found"}, status_code=404)
    img_path = session.page_images[page_num - 1]
    return FileResponse(str(img_path), media_type="image/png")


@app.get("/api/region-image/{session_id}/{page_num}/{coords}")
async def region_image(session_id: str, page_num: int, coords: str):
    """Crop a normalized 0..1000 OCR bounding box for figures and charts."""
    session = sessions.get(session_id)
    if not session or page_num < 1 or page_num > session.total_pages:
        return JSONResponse({"error": "Not found"}, status_code=404)
    try:
        box = [int(v) for v in coords.split(",")]
        if len(box) != 4:
            raise ValueError
    except ValueError:
        return JSONResponse({"error": "Invalid region"}, status_code=400)

    from PIL import Image

    image_path = session.page_images[page_num - 1]
    region_dir = session.upload_dir / "regions"
    region_dir.mkdir(exist_ok=True)
    output = region_dir / f"p{page_num}_{'_'.join(map(str, box))}.png"
    if not output.exists():
        with Image.open(image_path) as image:
            width, height = image.size
            x1, y1, x2, y2 = box
            pixel_box = (
                max(0, int(x1 / 1000 * width)), max(0, int(y1 / 1000 * height)),
                min(width, int(x2 / 1000 * width)), min(height, int(y2 / 1000 * height)),
            )
            image.crop(pixel_box).save(output, "PNG")
    return FileResponse(output, media_type="image/png")


@app.put("/api/edit")
async def edit(request: Request):
    body = await request.json()
    session_id = body.get("session_id")
    page_num = body.get("page_num")
    det_idx = body.get("detection_index")
    new_text = body.get("new_text", "")

    session = sessions.get(session_id)
    if not session:
        return JSONResponse({"error": "Session not found"}, status_code=404)

    result = session.page_results.get(page_num)
    if not result:
        return JSONResponse({"error": "Page not scanned"}, status_code=404)

    detections = result.get("detections", [])
    if 0 <= det_idx < len(detections):
        detections[det_idx]["text"] = new_text
        # Rebuild HTML from detections (not blocks) so indices stay 1:1
        html = ocr_parser.generate_html(detections, page_num)
        blocks = ocr_parser.reconstruct_structure(detections)
        result["html"] = html
        result["blocks"] = blocks

    return {"ok": True}


@app.post("/api/translate")
async def translate(request: Request):
    body = await request.json()
    session_id = body.get("session_id")
    page_num = body.get("page_num")
    source_lang = body.get("source_lang", "auto")
    target_lang = body.get("target_lang", "zh-CN")
    detections = body.get("detections", [])

    try:
        translated = await asyncio.to_thread(
            translator.translate_page, detections, source_lang, target_lang
        )
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

    # Store translation in session
    session = sessions.get(session_id)
    if session:
        session.page_translations[page_num] = translated

    return {"page_num": page_num, "translated_detections": translated}


@app.post("/api/export")
async def export_docx(request: Request):
    body = await request.json()
    session_id = body.get("session_id")
    export_mode = body.get("export_mode", "original")  # "original" | "translated" | "bilingual"

    session = sessions.get(session_id)
    if not session:
        return JSONResponse({"error": "Session not found"}, status_code=404)

    pages = []
    for page_num in sorted(session.page_results.keys()):
        result = session.page_results[page_num]
        page_data = {
            "page_num": page_num,
            "blocks": result.get("blocks", []),
            "detections": result.get("detections", []),
            "page_image": str(session.page_images[page_num - 1]),
        }
        for det in page_data["detections"]:
            if det.get("type") in {"image", "figure", "chart", "diagram"}:
                bbox = det.get("bbox") or []
                if len(bbox) >= 4:
                    try:
                        from PIL import Image
                        with Image.open(session.page_images[page_num - 1]) as image:
                            w, h = image.size
                            box = (
                                max(0, min(w, int(bbox[0] * w / 1000))),
                                max(0, min(h, int(bbox[1] * h / 1000))),
                                max(0, min(w, int(bbox[2] * w / 1000))),
                                max(0, min(h, int(bbox[3] * h / 1000))),
                            )
                            crop = session.upload_dir / f"word-image-{page_num}-{bbox[0]}-{bbox[1]}-{bbox[2]}-{bbox[3]}.png"
                            image.crop(box).save(crop, "PNG")
                            det["image_path"] = str(crop)
                    except Exception:
                        pass
        if page_num in session.page_translations:
            page_data["translations"] = session.page_translations[page_num]
        pages.append(page_data)

    output_path = session.upload_dir / f"{session.source_name}_ocr.docx"
    await asyncio.to_thread(docx_exporter.export_docx, pages, output_path, export_mode)

    return FileResponse(
        str(output_path),
        media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        filename=output_path.name,
    )


@app.post("/api/export-markdown")
async def export_markdown(request: Request):
    body = await request.json()
    session = sessions.get(body.get("session_id"))
    if not session:
        return JSONResponse({"error": "Session not found"}, status_code=404)

    pages = []
    for page_num in sorted(session.page_results):
        markdown = session.page_results[page_num].get("markdown", "").strip()
        markdown = _materialize_markdown_images(markdown, session, page_num)
        if session.total_pages > 1:
            pages.append(f"<!-- 第 {page_num} 页 -->\n\n{markdown}")
        else:
            pages.append(markdown)
    stem = Path(session.source_name).stem
    output_path = session.upload_dir / f"{stem}_ocr.md"
    output_path.write_text("\n\n---\n\n".join(pages) + "\n", encoding="utf-8")
    bundle_path = session.upload_dir / f"{stem}_ocr_markdown.zip"
    with zipfile.ZipFile(bundle_path, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.write(output_path, output_path.name)
        image_dir = session.upload_dir / "images"
        if image_dir.exists():
            for image_path in sorted(image_dir.glob("*.png")):
                bundle.write(image_path, f"images/{image_path.name}")
    return FileResponse(bundle_path, media_type="application/zip", filename=bundle_path.name)


def _materialize_markdown_images(markdown: str, session: SessionData, page_num: int) -> str:
    """Copy region-image references beside the exported Markdown file."""
    pattern = re.compile(r"!\[([^]]*)\]\(/api/region-image/[^/]+/(\d+)/([^)]*)\)")
    image_dir = session.upload_dir / "images"
    image_dir.mkdir(exist_ok=True)

    def replace(match):
        alt, pnum, coords = match.groups()
        coords = unquote(coords)
        src = session.page_images[int(pnum) - 1]
        try:
            from PIL import Image
            x1, y1, x2, y2 = [int(float(x.strip())) for x in coords.split(",")]
            with Image.open(src) as image:
                w, h = image.size
                # Detection boxes are in the model's 1000x1000 coordinate space.
                box = (max(0, int(x1*w/1000)), max(0, int(y1*h/1000)), min(w, int(x2*w/1000)), min(h, int(y2*h/1000)))
                name = f"page-{pnum}-{x1}-{y1}-{x2}-{y2}.png"
                image.crop(box).save(image_dir / name, "PNG")
            return f"![{alt}](images/{name})"
        except Exception:
            return match.group(0)
    return pattern.sub(replace, markdown)


# ── Run ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host=HOST, port=PORT, reload=False, log_level="info")
