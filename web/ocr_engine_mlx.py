"""Fast Apple-Silicon OCR engine powered by MLX."""
from __future__ import annotations

import threading
import time
from pathlib import Path

from config import OCR_REPO_DIR

MODEL_DIR = Path(
    __import__("os").environ.get(
        "OCR_MLX_MODEL_DIR", str(OCR_REPO_DIR.parent / "Unlimited-OCR-MLX")
    )
).expanduser().resolve()
ORIGINAL_TOKENIZER = OCR_REPO_DIR.parent / "Unlimited-OCR-model" / "tokenizer.json"

_model = None
_processor = None
_decoder = None
_lock = threading.Lock()
_inference_lock = threading.Lock()
_load_time = None
_last_stats = None


def ensure_model():
    """Load the quantized model once and keep it resident in unified memory."""
    global _model, _processor, _decoder, _load_time
    if _model is not None:
        return _model, _processor, _decoder

    with _lock:
        if _model is not None:
            return _model, _processor, _decoder
        if not (MODEL_DIR / "model.safetensors").exists():
            raise FileNotFoundError(f"MLX model not found: {MODEL_DIR}")
        if not ORIGINAL_TOKENIZER.exists():
            raise FileNotFoundError(f"Original tokenizer not found: {ORIGINAL_TOKENIZER}")

        started_at = time.time()
        from mlx_vlm import load
        from transformers import PreTrainedTokenizerFast

        _model, _processor = load(str(MODEL_DIR))

        # The community conversion's input tokenizer is required by its processor,
        # but its decoder loses byte-level spaces/newlines. Decode generated IDs
        # with Baidu's original tokenizer.json to preserve exact OCR text.
        _decoder = PreTrainedTokenizerFast(
            tokenizer_file=str(ORIGINAL_TOKENIZER),
            bos_token="<｜begin▁of▁sentence｜>",
            eos_token="<｜end▁of▁sentence｜>",
            pad_token="<｜▁pad▁｜>",
        )
        _load_time = time.time() - started_at
        print(f"[ocr_engine_mlx] model loaded in {_load_time:.1f}s")

    return _model, _processor, _decoder


def get_status():
    return {
        "loaded": _model is not None,
        "device": "mlx-metal" if _model is not None else None,
        "backend": "mlx-mxfp8",
        "load_time_s": round(_load_time, 1) if _load_time else None,
        "last_inference": _last_stats,
    }


def _clean_generated_text(decoder, token_ids: list[int]) -> str:
    text = decoder.decode(token_ids, skip_special_tokens=False)
    text = text.replace("<｜end▁of▁sentence｜>", "").strip()
    first_detection = text.find("<|det|>")
    if first_detection > 0:
        text = text[first_detection:]
    return text


def ocr_page_stream(image_path: str, max_length: int = 2048):
    """Yield progressively decoded OCR snapshots and a final statistics item."""
    global _last_stats
    model, processor, decoder = ensure_model()

    from mlx_vlm import stream_generate

    token_ids = []
    last = None
    started_at = time.time()
    max_tokens = max(128, min(int(max_length), 4096))

    # A single model instance must not run two Metal generations concurrently.
    with _inference_lock:
        for response in stream_generate(
            model,
            processor,
            "<image><|grounding|>Convert the document to markdown.",
            image=str(image_path),
            max_tokens=max_tokens,
            temperature=0.0,
        ):
            last = response
            if response.token is not None:
                token_ids.append(response.token)
            # Frequent small snapshots keep pause latency below a fraction of a
            # second. The browser animates the delta one Unicode character at a
            # time, avoiding the network cost of sending the full text per token.
            if token_ids and len(token_ids) % 4 == 0:
                yield {
                    "text": _clean_generated_text(decoder, token_ids),
                    "tokens": len(token_ids),
                    "done": False,
                }

    text = _clean_generated_text(decoder, token_ids)
    elapsed = time.time() - started_at
    _last_stats = {
        "seconds": round(elapsed, 2),
        "tokens": len(token_ids),
        "tokens_per_second": round(last.generation_tps, 1) if last else None,
        "peak_memory_gb": round(last.peak_memory, 2) if last else None,
    }
    print(f"[ocr_engine_mlx] page finished: {_last_stats}")
    yield {"text": text, "tokens": len(token_ids), "done": True, "stats": _last_stats}


def ocr_page(image_path: str, max_length: int = 2048) -> str:
    """Compatibility wrapper returning only the final page text."""
    final_text = ""
    for item in ocr_page_stream(image_path, max_length):
        final_text = item["text"]
    return final_text
