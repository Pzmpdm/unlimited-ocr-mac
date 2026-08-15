# Issue 3: page failure state + page-level retry (shared process_page).
import httpx
import pytest

from server import process_page, sessions


async def collect_events(session, engine, page_nums, **kw):
    events = []
    for p in page_nums:
        async for ev in process_page(session, p, engine=engine, **kw):
            events.append(ev)
    return events


def _text_det(text):
    return f"<|det|>text [1,2,30,40]<|/det|>{text}"


@pytest.mark.asyncio
async def test_page_status_starts_pending(session_factory):
    session = session_factory(n=3)
    assert [session.page_status[i]["state"] for i in (1, 2, 3)] == ["pending", "pending", "pending"]


@pytest.mark.asyncio
async def test_success_page_becomes_done(session_factory, fake_ocr):
    session = session_factory(n=1)
    fake_ocr.page_texts[str(session.page_images[0])] = _text_det("hello")
    events = [ev async for ev in process_page(session, 1, engine=fake_ocr)]
    assert session.page_status[1]["state"] == "done"
    assert session.page_status[1]["attempts"] == 1
    assert session.page_results[1]["detections"][0]["text"] == "hello"
    names = [ev["event"] for ev in events]
    assert names[0] == "page_start"
    assert "page_done" in names
    assert names[-1] == "page_image"


@pytest.mark.asyncio
async def test_failed_page_does_not_abort_scan(session_factory, fake_ocr):
    session = session_factory(n=3)
    imgs = [str(p) for p in session.page_images]
    fake_ocr.page_texts = {imgs[0]: _text_det("one"), imgs[2]: _text_det("three")}
    fake_ocr.fail_pages = {imgs[1]: RuntimeError("Metal out of memory")}
    events = await collect_events(session, fake_ocr, [1, 2, 3])
    assert session.page_status[1]["state"] == "done"
    assert session.page_status[2]["state"] == "failed"
    assert session.page_status[2]["error"] == "Metal out of memory"
    assert session.page_status[2]["attempts"] == 1
    assert session.page_status[3]["state"] == "done"
    assert any(ev["event"] == "error" and ev["data"]["page_num"] == 2 for ev in events)
    assert any(ev["event"] == "page_done" and ev["data"]["page_num"] == 3 for ev in events)


@pytest.mark.asyncio
async def test_retry_failed_page_succeeds(session_factory, fake_ocr):
    session = session_factory(n=1)
    img = str(session.page_images[0])
    fake_ocr.page_texts = {img: _text_det("fixed")}
    fake_ocr.fail_pages = {img: RuntimeError("boom")}
    await collect_events(session, fake_ocr, [1])
    assert session.page_status[1]["state"] == "failed"
    assert session.page_status[1]["attempts"] == 1
    # Fix the engine and retry the same page.
    fake_ocr.fail_pages.pop(img)
    await collect_events(session, fake_ocr, [1])
    assert session.page_status[1]["state"] == "done"
    assert session.page_status[1]["attempts"] == 2
    assert session.page_status[1]["error"] is None
    assert session.page_results[1]["status"] == "done"
    assert session.page_results[1]["detections"][0]["text"] == "fixed"


@pytest.mark.asyncio
async def test_retry_does_not_touch_other_pages(session_factory, fake_ocr):
    session = session_factory(n=3)
    imgs = [str(p) for p in session.page_images]
    fake_ocr.page_texts = {imgs[0]: _text_det("one"), imgs[1]: _text_det("two"), imgs[2]: _text_det("three")}
    await collect_events(session, fake_ocr, [1, 2, 3])
    p1_before = dict(session.page_results[1])
    engine2 = type(fake_ocr)(page_texts={imgs[1]: _text_det("RETRIED")})
    await collect_events(session, engine2, [2])
    assert session.page_results[1] == p1_before
    assert session.page_results[3]["detections"][0]["text"] == "three"
    assert session.page_results[2]["detections"][0]["text"] == "RETRIED"


@pytest.mark.asyncio
async def test_truncated_page_becomes_warning(session_factory, fake_ocr):
    session = session_factory(n=1)
    img = str(session.page_images[0])
    fake_ocr.streams[img] = [{
        "text": "<|det|>text [1,2,30,40]<|/det|>hello",
        "tokens": 6, "done": True, "truncated": True,
        "stats": {"max_tokens": 128, "truncated": True},
    }]
    await collect_events(session, fake_ocr, [1])
    assert session.page_status[1]["state"] == "warning"
    assert session.page_results[1]["truncated"] is True


@pytest.mark.asyncio
async def test_force_mode_vlm_runs_engine_on_native_page(session_factory, fake_ocr):
    session = session_factory(n=1, native_texts=["plenty of native text for the router"])
    fake_ocr.page_texts[str(session.page_images[0])] = _text_det("forced")
    await collect_events(session, fake_ocr, [1], force_mode="vlm")
    assert fake_ocr.calls and fake_ocr.calls[0][0] in ("ocr_page", "ocr_page_stream")
    assert session.page_results[1]["source"] == "vlm"


@pytest.mark.asyncio
async def test_scan_page_unknown_session_returns_404():
    from server import app

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        r = await client.post("/api/scan-page/nope/1", json={})
    assert r.status_code == 404


@pytest.mark.asyncio
async def test_scan_page_out_of_range_returns_404(session_factory):
    from server import app

    session = session_factory(n=2)
    sessions[session.session_id] = session
    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            r = await client.post(f"/api/scan-page/{session.session_id}/99", json={})
        assert r.status_code == 404
    finally:
        sessions.pop(session.session_id, None)
