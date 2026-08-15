# Issue 1: multiline detection parsing + strict bbox validation.
import pytest

from ocr_parser import (
    parse_ocr_output,
    normalize_bbox,
    raw_to_markdown,
    reconstruct_structure,
    generate_html,
)


def test_empty_raw_returns_empty_list():
    assert parse_ocr_output("") == []
    assert parse_ocr_output(None) == []


def test_single_line_text_detection():
    raw = "<|det|>text [1,2,30,40]<|/det|>hello"
    assert parse_ocr_output(raw) == [
        {"type": "text", "bbox": [1, 2, 30, 40], "text": "hello"},
    ]


def test_multiline_table_is_one_detection():
    raw = "<|det|>table [10,20,900,500]<|/det|>|A|B|\n|-|-|\n|1|2|"
    dets = parse_ocr_output(raw)
    assert len(dets) == 1
    assert dets[0]["type"] == "table"
    assert dets[0]["bbox"] == [10, 20, 900, 500]
    assert dets[0]["text"] == "|A|B|\n|-|-|\n|1|2|"


def test_multiple_detections_adjacent_without_newlines():
    raw = (
        "<|det|>text [1,2,30,40]<|/det|>first"
        "<|det|>text [50,60,80,90]<|/det|>second"
    )
    dets = parse_ocr_output(raw)
    assert len(dets) == 2
    assert [d["text"] for d in dets] == ["first", "second"]
    assert dets[1]["bbox"] == [50, 60, 80, 90]


def test_multiline_formula_detection():
    raw = (
        "<|det|>display-formula [1,1,10,10]<|/det|>E = mc^2\n"
        "a + b = c\n"
    )
    dets = parse_ocr_output(raw)
    assert len(dets) == 1
    assert dets[0]["type"] == "display-formula"
    assert dets[0]["bbox"] == [1, 1, 10, 10]
    assert dets[0]["text"] == "E = mc^2\na + b = c"


def test_malformed_bbox_keeps_text():
    raw = "<|det|>text [a,b,c,d]<|/det|>keep me"
    dets = parse_ocr_output(raw)
    assert len(dets) == 1
    assert dets[0]["bbox"] == []
    assert dets[0]["text"] == "keep me"


def test_out_of_range_bbox_returns_empty():
    raw = "<|det|>text [2000,1,3000,2]<|/det|>x"
    assert parse_ocr_output(raw)[0]["bbox"] == []


def test_inverted_bbox_returns_empty():
    raw = "<|det|>text [50,50,10,10]<|/det|>x"
    assert parse_ocr_output(raw)[0]["bbox"] == []


def test_truncated_record_does_not_crash():
    raw = (
        "<|det|>text [1,2,30,40]<|/det|>hello\n"
        "<|det|>table [5,6,7,8]"  # no closing tag
    )
    dets = parse_ocr_output(raw)
    assert len(dets) == 1
    assert dets[0]["text"] == "hello"
    # A raw string that is only a truncated record yields nothing, no crash.
    assert parse_ocr_output("<|det|>table [1,2,3,4]") == []


def test_bare_text_fallback_without_duplication():
    raw = (
        "preface garbage\n"
        "<|det|>text [1,2,30,40]<|/det|>hello\n\n"
        "trailing text"
    )
    dets = parse_ocr_output(raw)
    texts = [d["text"] for d in dets]
    assert "preface garbage" in texts
    assert "hello" in texts
    assert "trailing text" in texts
    # Detection content appears exactly once.
    assert texts.count("hello") == 1


def test_bare_text_between_records_is_fallback():
    raw = (
        "<|det|>text [1,2,30,40]<|/det|>first\n\n"
        "middle text\n\n"
        "<|det|>text [50,60,70,80]<|/det|>second"
    )
    dets = parse_ocr_output(raw)
    assert [d["text"] for d in dets] == ["first", "middle text", "second"]


def test_artifact_lines_are_filtered():
    raw = (
        "[non-text]\n"
        "<|det|>text [1,2,30,40]<|/det|>real\n\n"
        "figure [1,2,3,4]"  # bare artifact after blank line
    )
    dets = parse_ocr_output(raw)
    assert [d["text"] for d in dets] == ["real"]


def test_normalize_bbox_unit():
    assert normalize_bbox([1, 2, 30, 40]) == [1, 2, 30, 40]
    assert normalize_bbox((1, 2, 30, 40)) == [1, 2, 30, 40]
    assert normalize_bbox([1, 2, 3]) == []
    assert normalize_bbox([1, 2, 3, 4, 5]) == []
    assert normalize_bbox(["a", "b", "c", "d"]) == []
    assert normalize_bbox([-1, 0, 10, 10]) == []
    assert normalize_bbox([1001, 0, 10, 10]) == []
    assert normalize_bbox([10, 10, 10, 20]) == []  # zero width
    assert normalize_bbox([0, 0, 1000, 1000]) == [0, 0, 1000, 1000]
    assert normalize_bbox(["33.5", "114", "761.2", "150"]) == [33, 114, 761, 150]


def test_whitespace_in_metadata_is_tolerated():
    raw = "<|det|> text [1, 2, 30, 40] <|/det|>hello"
    dets = parse_ocr_output(raw)
    assert dets[0]["type"] == "text"
    assert dets[0]["bbox"] == [1, 2, 30, 40]


def test_raw_to_markdown_still_works_with_new_types():
    raw = "<|det|>display-formula [1,1,10,10]<|/det|>E=mc^2\n<|det|>text [1,2,3,4]<|/det|>body"
    md = raw_to_markdown(raw)
    assert "E=mc^2" in md
    assert "body" in md
    assert "<|det|>" not in md


def test_reconstruct_structure_and_html_still_work():
    dets = parse_ocr_output(
        "<|det|>title [35, 38, 685, 88]<|/det|>Heading\n"
        "<|det|>table [10,20,900,500]<|/det|>|A|B|\n|-|-|\n|1|2|"
    )
    blocks = reconstruct_structure(dets)
    assert blocks[0]["block_type"] == "heading"
    assert any(b["block_type"] == "table" for b in blocks)
    html = generate_html(dets, 1)
    assert "data-detection-index=" in html
