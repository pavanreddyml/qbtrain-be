"""
Dataset download manager (Hugging Face + Kaggle), mirroring the HF/Ollama model
clients. Used by the shared DatasetSelector in the frontend.

  * list_installed(provider)  -> datasets already cached locally
  * request_download(...)     -> background-thread download (HF snapshot / kagglehub)
  * download_status()         -> progress for the in-flight / last download

Self-contained so apps (e.g. poisoneddataset) can offer dataset selection without
a bespoke endpoint. No model/LLM dependencies.
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# A small curated catalog per provider (the datasets the poisoned-dataset app
# uses). Arbitrary ids are still accepted by request_download().
# The qbtrain "database" datasets (images + descriptions + a backdoor_responses.json
# poison pool) pushed by the notebook. The poisoned training set is built from
# these at runtime per the user's params — no separate poisoned dataset needed.
DATASET_CATALOG: Dict[str, List[Dict[str, str]]] = {
    "huggingface": [
        {"id": "qbtrain/flowers-102-captions-db", "label": "Flowers-102 captions (caption) — qbtrain DB"},
        {"id": "qbtrain/brain-tumor-mri-db", "label": "Brain Tumor MRI (medical) — qbtrain DB"},
        {"id": "qbtrain/stock-chart-patterns-db", "label": "Stock chart patterns (finance) — qbtrain DB"},
    ],
    "kaggle": [
        {"id": "mustaphaelbakai/stock-chart-patterns", "label": "Stock chart patterns (raw, finance)"},
    ],
}

PROVIDERS = ["huggingface", "kaggle"]


# ============================================================
# Download state
# ============================================================
@dataclass
class _DLState:
    key: str
    provider: str
    dataset_id: str
    status: str = "idle"      # idle | downloading | done | error
    progress: float = 0.0
    detail: str = ""
    error: Optional[str] = None
    started_at: Optional[float] = None
    finished_at: Optional[float] = None


_LOCK = threading.Lock()
_STATE: Dict[str, _DLState] = {}


def _key(provider: str, dataset_id: str) -> str:
    return f"{provider}:{dataset_id}"


def _state_dict(s: _DLState) -> Dict[str, Any]:
    return {
        "key": s.key, "provider": s.provider, "dataset_id": s.dataset_id,
        "status": s.status, "progress": round(s.progress, 3), "detail": s.detail,
        "error": s.error, "started_at": s.started_at, "finished_at": s.finished_at,
    }


# ============================================================
# "Installed?" detection
# ============================================================
def _hf_installed() -> List[str]:
    try:
        from huggingface_hub import scan_cache_dir
        out = []
        for repo in scan_cache_dir().repos:
            if getattr(repo, "repo_type", None) == "dataset":
                out.append(repo.repo_id)
        return sorted(set(out))
    except Exception:
        return []


def _kaggle_cache_root() -> str:
    return os.path.join(os.path.expanduser("~"), ".cache", "kagglehub", "datasets")


def _kaggle_installed() -> List[str]:
    root = _kaggle_cache_root()
    found: List[str] = []
    if not os.path.isdir(root):
        return found
    for owner in os.listdir(root):
        odir = os.path.join(root, owner)
        if not os.path.isdir(odir):
            continue
        for name in os.listdir(odir):
            if os.path.isdir(os.path.join(odir, name)):
                found.append(f"{owner}/{name}")
    return sorted(set(found))


def list_installed(provider: str) -> List[str]:
    if provider == "huggingface":
        return _hf_installed()
    if provider == "kaggle":
        return _kaggle_installed()
    raise ValueError(f"unknown provider: {provider}")


def is_installed(provider: str, dataset_id: str) -> bool:
    installed = list_installed(provider)
    return any(dataset_id == m or dataset_id.replace("/", "__") == m for m in installed)


# ============================================================
# Download workers
# ============================================================
def _run_hf(state: _DLState, config: Optional[str]) -> None:
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        state.status = "error"; state.error = f"huggingface_hub not installed: {exc}"
        state.finished_at = time.time(); return
    try:
        state.status = "downloading"; state.progress = 0.1; state.detail = "resolving snapshot"
        path = snapshot_download(repo_id=state.dataset_id, repo_type="dataset")
        state.progress = 1.0; state.status = "done"; state.detail = f"cached at {path}"
    except Exception as exc:
        state.status = "error"; state.error = str(exc); state.progress = 0.0
    finally:
        state.finished_at = time.time()


def _run_kaggle(state: _DLState) -> None:
    try:
        import kagglehub
    except Exception:
        try:
            import subprocess
            subprocess.run(["pip", "install", "-q", "kagglehub"], check=True)
            import kagglehub  # noqa: F811
        except Exception as exc:
            state.status = "error"; state.error = f"kagglehub unavailable: {exc}"
            state.finished_at = time.time(); return
    if not (os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json"))
            or (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"))):
        state.status = "error"
        state.error = ("Kaggle credentials missing. Put kaggle.json at ~/.kaggle/kaggle.json "
                       "(chmod 600) or set KAGGLE_USERNAME / KAGGLE_KEY.")
        state.finished_at = time.time(); return
    try:
        state.status = "downloading"; state.progress = 0.1; state.detail = "contacting kaggle"
        path = kagglehub.dataset_download(state.dataset_id)
        state.progress = 1.0; state.status = "done"; state.detail = f"cached at {path}"
    except Exception as exc:
        state.status = "error"; state.error = str(exc); state.progress = 0.0
    finally:
        state.finished_at = time.time()


def request_download(provider: str, dataset_id: str, config: Optional[str] = None) -> Dict[str, Any]:
    if provider not in PROVIDERS:
        raise ValueError(f"unknown provider: {provider}")
    if not dataset_id:
        raise ValueError("dataset_id is required")
    key = _key(provider, dataset_id)
    with _LOCK:
        existing = _STATE.get(key)
        if existing and existing.status == "downloading":
            return _state_dict(existing)
        if is_installed(provider, dataset_id):
            s = _DLState(key=key, provider=provider, dataset_id=dataset_id,
                         status="done", progress=1.0, detail="already cached")
            _STATE[key] = s
            return _state_dict(s)
        s = _DLState(key=key, provider=provider, dataset_id=dataset_id,
                     status="downloading", started_at=time.time(), progress=0.02)
        _STATE[key] = s
        target = _run_hf if provider == "huggingface" else _run_kaggle
        args = (s, config) if provider == "huggingface" else (s,)
        threading.Thread(target=target, args=args, daemon=True).start()
    return _state_dict(s)


def download_status(provider: Optional[str] = None, dataset_id: Optional[str] = None) -> Dict[str, Any]:
    if provider and dataset_id:
        s = _STATE.get(_key(provider, dataset_id))
        if s is None:
            done = is_installed(provider, dataset_id)
            return {"key": _key(provider, dataset_id), "provider": provider,
                    "dataset_id": dataset_id, "status": "done" if done else "idle",
                    "progress": 1.0 if done else 0.0, "detail": "", "error": None}
        return _state_dict(s)
    # else return the most recent in-flight / last state
    if not _STATE:
        return {"status": "idle", "progress": 0.0}
    latest = max(_STATE.values(), key=lambda s: (s.started_at or 0))
    return _state_dict(latest)
