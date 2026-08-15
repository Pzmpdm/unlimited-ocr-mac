# Issue 2: configurable token cap + truncation reporting.
import pytest

from config import OCR_MAX_TOKENS
from ocr_engine_mlx import resolve_max_tokens, compute_truncation


def test_request_at_config_cap_passes_through():
    assert resolve_max_tokens(8192, 8192) == 8192


def test_request_above_config_cap_is_clamped():
    assert resolve_max_tokens(16000, 8192) == 8192
    assert resolve_max_tokens(12000, OCR_MAX_TOKENS) == OCR_MAX_TOKENS


def test_request_below_cap_stays_as_requested():
    assert resolve_max_tokens(512, 8192) == 512


def test_request_is_never_below_128():
    assert resolve_max_tokens(10, 8192) == 128


def test_hitting_cap_is_truncated_conservatively():
    # No stop reason exposed -> conservative len >= max check.
    assert compute_truncation([1] * 8192, 8192, stop_reason=None) is True
    assert compute_truncation([1] * 8191, 8192, stop_reason=None) is False


def test_stop_reason_length_means_truncated():
    assert compute_truncation([1] * 500, 8192, stop_reason="length") is True


def test_stop_reason_eos_means_not_truncated():
    assert compute_truncation([1] * 500, 8192, stop_reason="eos_token_id") is False
    assert compute_truncation([1] * 8192, 8192, stop_reason="eos_token_id") is False


def test_default_config_is_8192():
    assert OCR_MAX_TOKENS == 8192
