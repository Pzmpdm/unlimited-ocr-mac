# Test foundation for the OCR parser. The full multiline/bbox matrix is
# added together with the parser rewrite; these cases already hold today.
from ocr_parser import parse_ocr_output


def test_empty_raw_returns_empty_list():
    assert parse_ocr_output("") == []
    assert parse_ocr_output(None) == []


def test_single_line_text_detection():
    raw = "<|det|>text [1,2,30,40]<|/det|>hello"
    assert parse_ocr_output(raw) == [
        {"type": "text", "bbox": [1, 2, 30, 40], "text": "hello"},
    ]


def test_multiple_detections_on_separate_lines():
    raw = (
        "<|det|>title [35, 38, 685, 88]<|/det|>Unlimited OCR Test\n"
        "<|det|>text [33, 114, 761, 150]<|/det|>Baidu released a model.\n"
    )
    dets = parse_ocr_output(raw)
    assert len(dets) == 2
    assert dets[0]["type"] == "title"
    assert dets[1]["type"] == "text"
    assert dets[1]["bbox"] == [33, 114, 761, 150]
