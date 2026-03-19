# qbtrain/ai/classifiers/injection_classifier.py
from __future__ import annotations

import importlib
import threading
from pathlib import Path
from typing import Any, Dict, List, Tuple

_LOCK = threading.Lock()
_PIPELINES: Dict[str, Any] = {}

INJECTION_CLASSIFIERS: Dict[str, Dict[str, Any]] = {
    "protectai/deberta-v3-base-prompt-injection": {
        "display_name": "ProtectAI DeBERTa v3 Prompt Injection",
        "labels_positive": ["INJECTION"],
    },
    "deepset/deberta-v3-base-injection": {
        "display_name": "Deepset DeBERTa v3 Injection",
        "labels_positive": ["INJECTION"],
    },
    "meta-llama/Prompt-Guard-86M": {
        "display_name": "Meta Llama Prompt Guard 86M",
        "labels_positive": ["INJECTION", "JAILBREAK"],
    },
}

DEFAULT_MODELS_DIR = "./hf_models"


def _model_local_dir(model_id: str, models_dir: str) -> Path:
    base = Path(models_dir)
    plain = base / model_id
    if plain.exists():
        return plain
    return base / model_id.replace("/", "__")


def list_classifiers(models_dir: str = DEFAULT_MODELS_DIR) -> List[Dict[str, Any]]:
    result = []
    for model_id, meta in INJECTION_CLASSIFIERS.items():
        local_dir = _model_local_dir(model_id, models_dir)
        result.append({
            "model_id": model_id,
            "display_name": meta["display_name"],
            "installed": local_dir.exists() and any(local_dir.iterdir()) if local_dir.exists() else False,
        })
    return result


def _get_pipeline(model_id: str, models_dir: str):
    key = f"{models_dir}::{model_id}"
    with _LOCK:
        if key in _PIPELINES:
            return _PIPELINES[key]

    tr = importlib.import_module("transformers")
    local_dir = _model_local_dir(model_id, models_dir)
    if not local_dir.exists():
        raise ValueError(f"Classifier model not installed: {model_id}")

    pipe = tr.pipeline("text-classification", model=str(local_dir), tokenizer=str(local_dir))

    with _LOCK:
        _PIPELINES[key] = pipe
    return pipe


def classify(model_id: str, text: str, models_dir: str = DEFAULT_MODELS_DIR) -> Tuple[bool, float]:
    """
    Classify text for prompt injection, splitting into overlapping chunks
    so that payloads beyond the model's 512-token window are still detected.

    Returns (is_injection, confidence).  If *any* chunk is flagged as
    injection the whole text is considered an injection attempt.
    """
    if model_id not in INJECTION_CLASSIFIERS:
        raise ValueError(f"Unknown classifier: {model_id}")

    meta = INJECTION_CLASSIFIERS[model_id]
    positive_labels = {lbl.upper() for lbl in meta["labels_positive"]}

    pipe = _get_pipeline(model_id, models_dir)
    tokenizer = pipe.tokenizer

    # Tokenize (no special tokens) to measure real length
    token_ids = tokenizer.encode(text, add_special_tokens=False)

    # Reserve 2 tokens for [CLS] + [SEP]
    max_chunk = 510
    overlap = 50
    step = max_chunk - overlap

    # Build text chunks
    if len(token_ids) <= max_chunk:
        chunks = [text]
    else:
        chunks = []
        for start in range(0, len(token_ids), step):
            chunk_ids = token_ids[start : start + max_chunk]
            chunks.append(tokenizer.decode(chunk_ids, skip_special_tokens=True))
            if start + max_chunk >= len(token_ids):
                break

    # Classify each chunk; short-circuit on first injection
    highest_confidence = 0.0
    for chunk in chunks:
        results = pipe(chunk, truncation=True, max_length=512)
        if not results:
            continue

        top = results[0]
        label = str(top.get("label", "")).upper()
        score = float(top.get("score", 0.0))

        is_injection = label in positive_labels
        confidence = score if is_injection else 1.0 - score

        if is_injection:
            return True, confidence

        highest_confidence = max(highest_confidence, confidence)

    return False, highest_confidence
