import os
import sys
from pathlib import Path

import pytest

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
if str(WEB_DIR) not in sys.path:
    sys.path.insert(0, str(WEB_DIR))

# Keep test runs deterministic: default MLX backend, but nothing loads yet.
os.environ.setdefault("OCR_BACKEND", "mlx")


class FakeOcrEngine:
    """In-memory OCR engine implementing the ocr_page / ocr_page_stream
    contract used by server.process_page. Records every invocation so tests
    can assert what was (or was not) sent to the visual model."""

    def __init__(self, page_texts=None, streams=None, fail_pages=None):
        # image_path (str) -> full raw OCR text (single-shot mode)
        self.page_texts = dict(page_texts or {})
        # image_path (str) -> list of stream items for streaming mode
        self.streams = dict(streams or {})
        # image_path (str) -> exception to raise on OCR
        self.fail_pages = dict(fail_pages or {})
        self.calls = []  # list of (mode, image_path, max_length)

    def ocr_page(self, image_path: str, max_length: int = 8192) -> str:
        image_path = str(image_path)
        self.calls.append(("ocr_page", image_path, max_length))
        if image_path in self.fail_pages:
            raise self.fail_pages[image_path]
        return self.page_texts.get(image_path, "")

    def ocr_page_stream(self, image_path: str, max_length: int = 8192):
        image_path = str(image_path)
        self.calls.append(("ocr_page_stream", image_path, max_length))
        if image_path in self.fail_pages:
            raise self.fail_pages[image_path]
        if image_path in self.streams:
            yield from self.streams[image_path]
            return
        text = self.page_texts.get(image_path, "")
        yield {
            "text": text,
            "tokens": len(text),
            "done": True,
            "truncated": False,
            "stats": {"seconds": 0.1, "tokens": len(text), "tokens_per_second": 10.0, "peak_memory_gb": 0.1},
        }


@pytest.fixture
def fake_ocr():
    """A fresh FakeOcrEngine with no pages configured."""
    return FakeOcrEngine()


@pytest.fixture
def sample_detection():
    return {"type": "text", "bbox": [1, 2, 30, 40], "text": "hello"}


@pytest.fixture
def session_factory(tmp_path):
    """Build a minimal SessionData (server.SessionData) for scan tests.

    Page images are tiny placeholder files; the FakeOcrEngine is configured per
    test with page_texts / fail_pages keyed by their str() paths.
    """

    def _make(n=3, native_texts=None, native_pages=None):
        from server import SessionData

        upload_dir = tmp_path / "uploads"
        upload_dir.mkdir(parents=True, exist_ok=True)
        page_images = [upload_dir / f"page_{i:04d}.png" for i in range(1, n + 1)]
        for p in page_images:
            p.write_bytes(b"fake-png")
        texts = native_texts if native_texts is not None else ["" for _ in range(n)]
        pages = native_pages if native_pages is not None else [
            {"text": texts[i], "markdown": "", "detections": [], "tables": 0, "figures": 0}
            for i in range(n)
        ]
        return SessionData(
            f"test-{Path(upload_dir).name}", upload_dir, "test.pdf", n, page_images, texts, pages
        )

    return _make
