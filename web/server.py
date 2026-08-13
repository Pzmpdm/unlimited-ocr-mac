"""Unlimited-OCR Web: FastAPI server with SSE streaming."""
from __future__ import annotations
import asyncio
import json
import os
import shutil
import time
import uuid
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

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
from config import PUBLIC_DIR, UPLOAD_DIR, PORT, HOST


# ── Session store ──────────────────────────────────────────────

class SessionData:
    def __init__(self, session_id: str, upload_dir: Path, source_name: str,
                 total_pages: int, page_images: list[Path]):
        self.session_id = session_id
        self.upload_dir = upload_dir
        self.source_name = source_name
        self.total_pages = total_pages
        self.page_images = page_images
        self.page_results: dict[int, dict] = {}  # page_num → {detections, html, raw, blocks}
        self.page_translations: dict[int, list] = {}
        self.created_at = time.time()


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
        # Convert PDF to PNGs
        page_images = pdf_converter.pdf_to_images(str(src_path), session_dir)
    elif ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp"):
        page_images = [src_path]
    else:
        shutil.rmtree(session_dir, ignore_errors=True)
        return JSONResponse({"error": f"Unsupported file type: {ext}"}, status_code=400)

    total_pages = len(page_images)
    sessions[session_id] = SessionData(session_id, session_dir, source_name, total_pages, page_images)

    return {
        "session_id": session_id,
        "total_pages": total_pages,
        "source_name": source_name,
    }


@app.post("/api/scan")
async def scan(request: Request):
    """SSE endpoint: stream generated tokens, then structured page results."""
    body = await request.json()
    session_id = body.get("session_id")
    max_length = body.get("max_length", 8192)

    session = sessions.get(session_id)
    if not session:
        return JSONResponse({"error": "Session not found"}, status_code=404)
    control = ScanControl()
    scan_controls[session_id] = control

    async def event_generator():
        stopped = False
        try:
            for page_num in range(1, session.total_pages + 1):
                if control.cancelled.is_set():
                    stopped = True
                    break
                yield _sse("page_start", {"page_num": page_num, "total_pages": session.total_pages})
                yield _sse("page_progress", {"page_num": page_num, "status": "scanning"})

                img_path = str(session.page_images[page_num - 1])

                try:
                    if hasattr(ocr_engine, "ocr_page_stream"):
                        stream = ocr_engine.ocr_page_stream(img_path, max_length)
                        raw = ""
                        while True:
                            while control.paused.is_set() and not control.cancelled.is_set():
                                await asyncio.sleep(0.08)
                            if control.cancelled.is_set():
                                stream.close()
                                stopped = True
                                break
                            done, item = await asyncio.to_thread(_next_stream_item, stream)
                            if done:
                                break
                            raw = item["text"]
                            yield _sse("token", {
                                "page_num": page_num,
                                "text": raw,
                                "markdown": ocr_parser.raw_to_markdown(raw, session_id, page_num),
                                "tokens": item.get("tokens", 0),
                                "done": item.get("done", False),
                                "stats": item.get("stats"),
                            })
                    else:
                        raw = await asyncio.to_thread(ocr_engine.ocr_page, img_path, max_length)
                except Exception as e:
                    if control.cancelled.is_set():
                        stopped = True
                        break
                    yield _sse("error", {"page_num": page_num, "message": str(e)})
                    continue

                if stopped:
                    break

                yield _sse("page_progress", {"page_num": page_num, "status": "parsing"})

                detections = ocr_parser.parse_ocr_output(raw)
                blocks = ocr_parser.reconstruct_structure(detections)

            # Store full results
                html = ocr_parser.generate_html(detections, page_num)
                markdown = ocr_parser.raw_to_markdown(raw, session_id, page_num)
                session.page_results[page_num] = {
                    "detections": detections, "html": html, "raw": raw,
                    "markdown": markdown, "blocks": blocks,
                }

            # Push each detection line one by one for real-time display
                for i, det in enumerate(detections):
                    det_html = ocr_parser.generate_det_html(det, i)
                    yield _sse("det_result", {
                        "page_num": page_num, "det_index": i,
                        "detection": det, "html": det_html,
                        "total_detections": len(detections),
                    })

                yield _sse("page_done", {"page_num": page_num, "html": html, "markdown": markdown})
                yield _sse("page_image", {"page_num": page_num, "image_url": f"/api/page-image/{session_id}/{page_num}"})

            yield _sse("scan_stopped" if stopped else "scan_complete", {
                "session_id": session_id, "total_pages": session.total_pages
            })
        finally:
            scan_controls.pop(session_id, None)

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
        }
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
        if session.total_pages > 1:
            pages.append(f"<!-- 第 {page_num} 页 -->\n\n{markdown}")
        else:
            pages.append(markdown)
    output_path = session.upload_dir / f"{Path(session.source_name).stem}_ocr.md"
    output_path.write_text("\n\n---\n\n".join(pages) + "\n", encoding="utf-8")
    return FileResponse(output_path, media_type="text/markdown; charset=utf-8", filename=output_path.name)


# ── Run ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host=HOST, port=PORT, reload=False, log_level="info")
