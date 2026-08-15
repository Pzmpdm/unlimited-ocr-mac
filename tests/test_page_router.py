# Issue 4: page-quality router (native / hybrid / vlm).
from page_router import (
    HYBRID_IMAGE_COVERAGE,
    MAX_SUSPICIOUS_CHAR_RATIO,
    MIN_NATIVE_TEXT_CHARS,
    PageAnalysis,
    choose_page_route,
    estimate_column_count,
    suspicious_character_ratio,
)


def test_dense_single_column_text_page_is_native():
    a = PageAnalysis(text_chars=3000, text_blocks=40, native_markdown_chars=2800)
    assert choose_page_route(a) == "native"


def test_sparse_text_page_is_vlm():
    assert choose_page_route(PageAnalysis(text_chars=10)) == "vlm"
    assert choose_page_route(PageAnalysis(text_chars=79)) == "vlm"


def test_table_page_is_hybrid():
    assert choose_page_route(PageAnalysis(text_chars=2000, table_count=3)) == "hybrid"


def test_image_heavy_page_is_hybrid():
    assert choose_page_route(PageAnalysis(text_chars=1500, image_coverage=0.4)) == "hybrid"


def test_suspicious_text_page_is_vlm():
    assert choose_page_route(PageAnalysis(text_chars=5000, suspicious_char_ratio=0.1)) == "vlm"


def test_two_column_page_is_hybrid():
    assert choose_page_route(PageAnalysis(text_chars=3000, column_count=2)) == "hybrid"


def test_boundary_thresholds():
    assert choose_page_route(PageAnalysis(text_chars=MIN_NATIVE_TEXT_CHARS)) == "native"
    assert choose_page_route(PageAnalysis(text_chars=80, image_coverage=HYBRID_IMAGE_COVERAGE - 0.001)) == "native"
    assert choose_page_route(PageAnalysis(text_chars=80, image_coverage=HYBRID_IMAGE_COVERAGE)) == "hybrid"
    assert choose_page_route(PageAnalysis(text_chars=80, suspicious_char_ratio=MAX_SUSPICIOUS_CHAR_RATIO + 0.01)) == "vlm"


def test_hybrid_requires_usable_text():
    # A page with a table but almost no text is still a scanned page -> vlm.
    assert choose_page_route(PageAnalysis(text_chars=10, table_count=3)) == "vlm"


def test_suspicious_character_ratio():
    assert suspicious_character_ratio("") == 0.0
    assert suspicious_character_ratio("plain ascii text") == 0.0
    assert suspicious_character_ratio("a\ufffdb\ufffdc") == 0.4
    assert suspicious_character_ratio("a\x01b") > 0.2  # control char
    assert suspicious_character_ratio("a\tb\nc\rd") == 0.0  # tab/nl/cr ok


def test_estimate_column_count():
    def lines(xs):
        return [{"bbox": (x, 100, x + 50, 110)} for x in xs]

    single = lines([10, 12, 15, 20, 25, 30, 40, 50])
    assert estimate_column_count(single) == 1
    two = lines([10, 12, 15, 18, 520, 525, 530, 540])
    assert estimate_column_count(two) == 2
    assert estimate_column_count([]) == 1
    assert estimate_column_count([{"bbox": (0, 0, 1, 1)}]) == 1


def test_from_dict_ignores_unknown_keys():
    a = PageAnalysis.from_dict({"text_chars": 123, "table_count": 2, "bogus": 1})
    assert a.text_chars == 123
    assert a.table_count == 2
    assert PageAnalysis.from_dict({}) == PageAnalysis()


def test_server_route_falls_back_without_analysis():
    # Legacy native pages (no analysis key) keep the text_chars heuristic.
    from server import _route_page

    assert _route_page({}, 3000, None) == "native"
    assert _route_page({}, 10, None) == "vlm"
    assert _route_page({}, 10, "native") == "native"
    assert _route_page({}, 3000, "vlm") == "vlm"
    assert _route_page({"analysis": {"text_chars": 2000, "table_count": 2}}, 0, None) == "hybrid"
