# Issue 5: region-level Hybrid OCR.
import pytest

from hybrid_ocr import (
    TABLE_OCR_THRESHOLD,
    assess_native_table_quality,
    build_hybrid_result,
    crop_page_region,
    map_region_bbox_to_page,
    merge_hybrid_regions,
)


def _good_table():
    return {
        "region_id": "table-1",
        "type": "table",
        "bbox": (100.0, 200.0, 900.0, 700.0),
        "normalized_bbox": [100, 200, 900, 700],
        "source": "native",
        "content": "|A|B|\n|-|-|\n|1|2|",
        "confidence": None,
    }


def _bad_table():
    return {
        "region_id": "table-2",
        "type": "table",
        "bbox": (100.0, 200.0, 900.0, 700.0),
        "normalized_bbox": [100, 200, 900, 700],
        "source": "native",
        "content": "|A|B|\n|1|",
        "confidence": None,
    }


def _text_region(i=1, y=50):
    return {
        "region_id": f"text-{i}",
        "type": "text",
        "bbox": (50.0, float(y), 600.0, float(y + 20)),
        "normalized_bbox": [50, y, 600, y + 20],
        "source": "native",
        "content": f"body line {i}",
        "confidence": None,
    }


def _native_page(regions):
    return {"text": "x", "markdown": "", "detections": [], "tables": 0, "figures": 0, "regions": regions}


def test_good_table_does_not_invoke_ocr(session_factory, fake_ocr):
    session = session_factory(n=1)
    page = _native_page([_text_region(), _good_table()])
    fake_ocr.page_texts[str(session.page_images[0])] = "<|det|>text [1,2,3,4]<|/det|>unused"
    result = build_hybrid_result(session, 1, page, fake_ocr, 8192)
    assert fake_ocr.calls == []
    assert result["source"] == "hybrid"
    assert result["warnings"] == []
    assert all(d["source"] == "native" for d in result["detections"])


def test_low_quality_table_triggers_region_ocr(session_factory, fake_ocr):
    session = session_factory(n=1)
    page = _native_page([_text_region(), _bad_table()])
    crop = str(session.upload_dir / "hybrid" / "table-2.png")
    fake_ocr.page_texts[crop] = "<|det|>table [0,0,1000,1000]<|/det|>|X|Y|\n|-|-|\n|1|2|"
    result = build_hybrid_result(session, 1, page, fake_ocr, 8192)
    assert len(fake_ocr.calls) == 1
    table_dets = [d for d in result["detections"] if d["type"] == "table"]
    assert table_dets
    assert table_dets[0]["source"] == "vlm"
    assert "|X|Y|" in table_dets[0]["text"]


def test_body_text_stays_native_not_duplicated(session_factory, fake_ocr):
    session = session_factory(n=1)
    page = _native_page([_text_region(1, 50), _text_region(2, 80)])
    result = build_hybrid_result(session, 1, page, fake_ocr, 8192)
    assert fake_ocr.calls == []
    texts = [d["text"] for d in result["detections"]]
    assert texts == ["body line 1", "body line 2"]
    assert all(d["source"] == "native" for d in result["detections"])


def test_region_ocr_failure_degrades_to_native(session_factory, fake_ocr):
    session = session_factory(n=1)
    page = _native_page([_bad_table()])
    fake_ocr.fail_pages[str(session.upload_dir / "hybrid" / "table-2.png")] = RuntimeError("boom")
    result = build_hybrid_result(session, 1, page, fake_ocr, 8192)
    assert result["warnings"][0]["type"] == "region_ocr_failed"
    table_dets = [d for d in result["detections"] if d["type"] == "table"]
    assert table_dets and table_dets[0]["source"] == "native"


def test_bbox_conversion_local_to_page():
    got = map_region_bbox_to_page([100, 100, 500, 300], [100, 200, 900, 800])
    assert got == [180, 260, 500, 380]
    assert map_region_bbox_to_page([0, 0, 1000, 1000], [0, 0, 1000, 1000]) == [0, 0, 1000, 1000]
    # identity mapping for a full-page region
    assert map_region_bbox_to_page([1, 1, 999, 999], [0, 0, 1000, 1000]) == [1, 1, 999, 999]
    assert map_region_bbox_to_page([1, 2, 3], [0, 0, 1000, 1000]) == []


def test_merge_replaces_by_region_id():
    regions = [{"region_id": "table-1", "type": "table", "bbox": (0, 0, 10, 10),
                "normalized_bbox": [0, 0, 10, 10], "content": "old"}]
    repl = [{"region_id": "table-1", "type": "table", "bbox": (0, 0, 10, 10),
             "normalized_bbox": [0, 0, 10, 10], "content": "new", "source": "vlm"}]
    merged = merge_hybrid_regions(regions, repl)
    assert len(merged) == 1
    assert merged[0]["content"] == "new"
    assert merged[0]["source"] == "vlm"


def test_merge_order_is_stable():
    regions = [
        {"region_id": "text-2", "type": "text", "bbox": (0, 80, 10, 90),
         "normalized_bbox": [0, 80, 10, 90], "content": "b"},
        {"region_id": "text-1", "type": "text", "bbox": (0, 50, 10, 60),
         "normalized_bbox": [0, 50, 10, 60], "content": "a"},
    ]
    merged = merge_hybrid_regions(regions, [])
    assert [r["region_id"] for r in merged] == ["text-1", "text-2"]


def test_quality_threshold_constant():
    assert 0 < TABLE_OCR_THRESHOLD < 1
    assert assess_native_table_quality(_good_table()) >= TABLE_OCR_THRESHOLD
    assert assess_native_table_quality(_bad_table()) < TABLE_OCR_THRESHOLD


def test_crop_page_region(tmp_path):
    from PIL import Image

    img = tmp_path / "page.png"
    Image.new("RGB", (100, 100), "white").save(img)
    out = tmp_path / "crop.png"
    crop_page_region(img, [0, 0, 500, 500], out, padding=0)
    with Image.open(out) as im:
        assert im.size == (50, 50)
    # padding clamps to the image bounds
    out2 = tmp_path / "crop2.png"
    crop_page_region(img, [0, 0, 1000, 1000], out2, padding=8)
    with Image.open(out2) as im:
        assert im.size == (100, 100)  # clamped to image bounds


@pytest.mark.asyncio
async def test_hybrid_route_streams_page(session_factory, fake_ocr):
    from server import process_page

    session = session_factory(n=1, native_texts=["lots of text on this digital page"])
    session.native_pages[0]["analysis"] = {"text_chars": 3000, "table_count": 1}
    session.native_pages[0]["regions"] = [_text_region(1, 50), _good_table()]
    events = [ev async for ev in process_page(session, 1, engine=fake_ocr)]
    assert session.page_results[1]["source"] == "hybrid"
    assert session.page_status[1]["state"] == "done"
    assert any(ev["event"] == "page_done" and ev["data"]["source"] == "hybrid" for ev in events)


@pytest.mark.asyncio
async def test_hybrid_route_warning_on_region_failure(session_factory, fake_ocr):
    from server import process_page

    session = session_factory(n=1, native_texts=["lots of text on this digital page"])
    session.native_pages[0]["analysis"] = {"text_chars": 3000, "table_count": 1}
    session.native_pages[0]["regions"] = [_bad_table()]
    fake_ocr.fail_pages[str(session.upload_dir / "hybrid" / "table-2.png")] = RuntimeError("oom")
    await process_page(session, 1, engine=fake_ocr).__anext__() and None  # drain start
    events = [ev async for ev in process_page(session, 1, engine=fake_ocr)]
    assert session.page_results[1]["warnings"]
    assert session.page_status[1]["state"] == "warning"
