# qbtrain/ai/llm/huggingface_client.py
from __future__ import annotations

import importlib
import json
import os
import shutil
import threading
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Deque, Dict, Generator, List, Optional, Type

from pydantic import BaseModel
from qbtrain.tracers import Tracer

from .base_llm_client import LLMClient, Message

MessageList = Optional[List[Message]]


def _lazy_import_hfhub():
    return importlib.import_module("huggingface_hub")


def _lazy_import_transformers():
    return importlib.import_module("transformers")


def _hf_guardrails(top_k: int, presence_penalty: float, frequency_penalty: float) -> None:
    if presence_penalty not in (None, 0.0):
        raise ValueError("Hugging Face local generation does not support presence_penalty.")
    if frequency_penalty not in (None, 0.0):
        raise ValueError("Hugging Face local generation does not support frequency_penalty.")
    if top_k is not None and top_k < 1:
        raise ValueError("top_k must be >= 1.")


@dataclass
class DownloadTask:
    model_id: str
    revision: Optional[str] = None
    local_dir: Optional[Path] = None
    status: str = "queued"  # queued | downloading | completed | failed
    progress: float = 0.0
    message: str = ""
    expected_bytes: int = 0
    downloaded_bytes: int = 0


class HuggingFaceClient(LLMClient):
    client_id = "huggingface"
    display_name = "Hugging Face (local)"
    param_display_names = {"models_dir": "Models Directory", "model": "Local model directory name"}

    _LOCK = threading.RLock()
    _QUEUE: Deque[DownloadTask] = deque()
    _CURRENT: Optional[DownloadTask] = None
    _WORKER: Optional[threading.Thread] = None

    def __init__(
        self,
        models_dir: str = "./hf_models",
        model: Optional[str] = None,
        *,
        system_prompt: Optional[str] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        temperature: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
    ):
        super().__init__(
            model=model,
            system_prompt=system_prompt,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            max_output_tokens=max_output_tokens,
        )
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self._pipelines: Dict[str, Any] = {}

    def warmup_async(self, *, preload_model: Optional[str] = None) -> None:
        def _warm():
            _lazy_import_hfhub()
            _lazy_import_transformers()
            if preload_model:
                try:
                    self._get_pipeline(preload_model)
                except Exception:
                    pass

        threading.Thread(target=_warm, daemon=True).start()

    # ---------- Download manager ----------
    @classmethod
    def _ensure_worker(cls):
        with cls._LOCK:
            if cls._WORKER is None or not cls._WORKER.is_alive():
                cls._WORKER = threading.Thread(target=cls._worker_loop, daemon=True)
                cls._WORKER.start()

    @classmethod
    def _worker_loop(cls):
        while True:
            with cls._LOCK:
                if not cls._QUEUE:
                    cls._CURRENT = None
                    break
                task = cls._QUEUE.popleft()
                cls._CURRENT = task
                task.status = "downloading"

            try:
                hfhub = _lazy_import_hfhub()
                api = hfhub.HfApi()
                snapshot_download = hfhub.snapshot_download

                info = api.model_info(task.model_id, revision=task.revision)
                exp = sum((s.size or 0) for s in info.siblings if getattr(s, "size", None) is not None)
                with cls._LOCK:
                    task.expected_bytes = exp

                local_dir = task.local_dir or (Path("./hf_models") / task.model_id.replace("/", "__"))
                local_dir.mkdir(parents=True, exist_ok=True)

                stop_flag = threading.Event()

                def monitor():
                    while not stop_flag.is_set():
                        size = 0
                        for root, _, files in os.walk(local_dir):
                            for fn in files:
                                fp = os.path.join(root, fn)
                                try:
                                    size += os.path.getsize(fp)
                                except OSError:
                                    pass
                        with cls._LOCK:
                            task.downloaded_bytes = size
                            if task.expected_bytes > 0:
                                task.progress = min(99.0, (size / task.expected_bytes) * 100.0)
                        time.sleep(0.5)

                mon = threading.Thread(target=monitor, daemon=True)
                mon.start()

                snapshot_download(
                    repo_id=task.model_id,
                    revision=task.revision,
                    local_dir=str(local_dir),
                    local_dir_use_symlinks=False,
                    allow_patterns=["*"],
                )

                stop_flag.set()
                mon.join(timeout=2.0)

                with cls._LOCK:
                    task.local_dir = local_dir
                    task.progress = 100.0
                    task.status = "completed"
            except Exception as e:
                with cls._LOCK:
                    task.status = "failed"
                    task.message = str(e)
            finally:
                with cls._LOCK:
                    cls._CURRENT = None

    @classmethod
    def request_download(cls, model_id: str, revision: Optional[str] = None, local_dir: Optional[str] = None) -> None:
        with cls._LOCK:
            cls._QUEUE.append(
                DownloadTask(model_id=model_id, revision=revision, local_dir=Path(local_dir) if local_dir else None)
            )
        cls._ensure_worker()

    @classmethod
    def download_status(cls) -> Dict[str, Any]:
        with cls._LOCK:
            queue_list = [{"model_id": t.model_id, "status": t.status, "progress": round(t.progress, 2)} for t in list(cls._QUEUE)]
            current = None
            if cls._CURRENT:
                current = {
                    "model_id": cls._CURRENT.model_id,
                    "status": cls._CURRENT.status,
                    "progress": round(cls._CURRENT.progress, 2),
                    "message": cls._CURRENT.message,
                }
            return {"current": current, "queue": queue_list}

    def list_models(self) -> List[str]:
        if not self.models_dir.exists():
            return []
        return [p.name for p in self.models_dir.iterdir() if p.is_dir()]

    def delete_model(self, local_name: str) -> None:
        path = self.models_dir / local_name
        if path.exists():
            shutil.rmtree(path)

    # ---------- Inference ----------
    def _resolve_local_dir(self, model: str) -> Path:
        p1 = self.models_dir / model
        p2 = self.models_dir / model.replace("/", "__")
        if p1.exists():
            return p1
        if p2.exists():
            return p2
        raise ValueError(f"Local model directory not found for '{model}' under {self.models_dir}")

    @staticmethod
    def _is_vision_model(local_dir: Path) -> bool:
        """Detect if the model is a vision-language model (e.g. LLaVA) by checking config."""
        config_path = local_dir / "config.json"
        if not config_path.exists():
            return False
        try:
            import json as _json
            with open(config_path) as f:
                cfg = _json.load(f)
            model_type = cfg.get("model_type", "").lower()
            architectures = [a.lower() for a in cfg.get("architectures", [])]
            vision_keywords = ("llava", "vision", "vit", "clip", "blip", "idefics", "paligemma", "qwen2_vl")
            if any(kw in model_type for kw in vision_keywords):
                return True
            if any(any(kw in a for kw in vision_keywords) for a in architectures):
                return True
        except Exception:
            pass
        return False

    def _get_pipeline(self, model: str):
        if not hasattr(self, "_pipelines"):
            self._pipelines = {}
        if model in self._pipelines:
            return self._pipelines[model]

        tr = _lazy_import_transformers()
        AutoTokenizer = tr.AutoTokenizer
        AutoModelForCausalLM = tr.AutoModelForCausalLM
        pipeline = tr.pipeline

        local_dir = self._resolve_local_dir(model)
        tok = AutoTokenizer.from_pretrained(local_dir)
        mdl = AutoModelForCausalLM.from_pretrained(local_dir)
        pipe = pipeline("text-generation", model=mdl, tokenizer=tok)

        self._pipelines[model] = pipe
        return pipe

    def _get_vision_model(self, model: str):
        """Load vision-language model + processor (e.g. LLaVA)."""
        if not hasattr(self, "_vision_models"):
            self._vision_models: Dict[str, Any] = {}
        if model in self._vision_models:
            return self._vision_models[model]

        tr = _lazy_import_transformers()
        local_dir = self._resolve_local_dir(model)

        # Try LlavaForConditionalGeneration first, then generic AutoModel
        processor = tr.AutoProcessor.from_pretrained(local_dir)
        mdl = None
        for cls_name in ("LlavaForConditionalGeneration", "LlavaNextForConditionalGeneration",
                         "AutoModelForVision2Seq"):
            cls = getattr(tr, cls_name, None)
            if cls is None:
                continue
            try:
                mdl = cls.from_pretrained(local_dir)
                break
            except Exception:
                continue

        if mdl is None:
            raise ValueError(
                f"Could not load vision model from '{local_dir}'. "
                "Ensure the model is a supported vision-language model (e.g. LLaVA)."
            )

        entry = {"processor": processor, "model": mdl}
        self._vision_models[model] = entry
        return entry

    @staticmethod
    def _prepare_image(image: Any):
        """Convert image bytes/path to PIL Image."""
        from PIL import Image as PILImage
        import io as _io
        if isinstance(image, PILImage.Image):
            return image
        if isinstance(image, bytes):
            return PILImage.open(_io.BytesIO(image)).convert("RGB")
        if isinstance(image, str):
            import base64 as _b64
            if image.startswith("data:"):
                image = image.split(",", 1)[1]
            try:
                raw = _b64.b64decode(image)
                return PILImage.open(_io.BytesIO(raw)).convert("RGB")
            except Exception:
                return PILImage.open(image).convert("RGB")
        raise ValueError(f"Unsupported image type: {type(image)}")

    def response(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        conversation_history: MessageList = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        temperature: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        tracer: Optional[Tracer] = None,
        image: Optional[Any] = None,
        *args: Any,
        **kwargs: Any,
    ) -> str:
        if not self.model:
            raise ValueError("model is required (pass in clientDetails.params.model).")

        eff_system = self._effective_param("system_prompt", system_prompt)
        eff_top_k = self._effective_param("top_k", top_k)
        eff_top_p = self._effective_param("top_p", top_p)
        eff_temp = self._effective_param("temperature", temperature)
        eff_pres = self._effective_param("presence_penalty", presence_penalty)
        eff_freq = self._effective_param("frequency_penalty", frequency_penalty)
        eff_max = self._effective_param("max_output_tokens", max_output_tokens)

        _hf_guardrails(eff_top_k, eff_pres, eff_freq)

        conversation_history = self.trim_conversation_history(conversation_history)

        gen_kwargs: Dict[str, Any] = {}
        if eff_max is not None:
            gen_kwargs["max_new_tokens"] = eff_max
        if eff_temp is not None:
            gen_kwargs["temperature"] = eff_temp
        if eff_top_p is not None:
            gen_kwargs["top_p"] = eff_top_p
        if eff_top_k is not None:
            gen_kwargs["top_k"] = eff_top_k
        if any(k in gen_kwargs for k in ("temperature", "top_p", "top_k")):
            gen_kwargs["do_sample"] = True

        trace_extra = kwargs.pop("_trace", None)

        # --- Vision-language model path ---
        local_dir = self._resolve_local_dir(self.model)
        if image is not None and self._is_vision_model(local_dir):
            pil_image = self._prepare_image(image)
            vm = self._get_vision_model(self.model)
            processor, mdl = vm["processor"], vm["model"]

            # Build prompt with <image> token for LLaVA-style models
            text_prompt = ""
            if eff_system:
                text_prompt += f"{eff_system}\n\n"
            text_prompt += f"USER: <image>\n{prompt}\nASSISTANT:"

            with self._timed() as t:
                inputs = processor(text=text_prompt, images=pil_image, return_tensors="pt")
                output_ids = mdl.generate(**inputs, **gen_kwargs)
                result = processor.decode(output_ids[0], skip_special_tokens=True)

            # Strip the prompt from the output if echoed
            if "ASSISTANT:" in result:
                result = result.split("ASSISTANT:")[-1].strip()

            self._trace(
                tracer,
                operation="response",
                model=self.model,
                params={k: v for k, v in {"temperature": eff_temp, "top_p": eff_top_p, "top_k": eff_top_k, "max_output_tokens": eff_max}.items() if v is not None},
                system_prompt_preview=(eff_system[:200] if eff_system else None),
                system_prompt_length=(len(eff_system) if eff_system else 0),
                prompt_preview=prompt[:200],
                prompt_length=len(prompt),
                conv_history_length=len(conversation_history or []),
                image_provided=True,
                latency_ms=t.ms,
                **(trace_extra or {}),
                input_tokens=None,
                output_tokens=None,
                total_tokens=None,
            )
            return result

        # --- Text-only path (original) ---
        full_prompt = ""
        if eff_system:
            full_prompt += f"{eff_system}\n\n"
        if conversation_history:
            for m in conversation_history:
                full_prompt += f"{m.get('role','user')}: {m.get('content','')}\n"
        full_prompt += f"user: {prompt}\nassistant:"

        pipe = self._get_pipeline(self.model)

        with self._timed() as t:
            out = pipe(full_prompt, **gen_kwargs)[0]["generated_text"]
        result = out.split("assistant:", 1)[-1].strip()

        self._trace(
            tracer,
            operation="response",
            model=self.model,
            params={k: v for k, v in {"temperature": eff_temp, "top_p": eff_top_p, "top_k": eff_top_k, "max_output_tokens": eff_max}.items() if v is not None},
            system_prompt_preview=(eff_system[:200] if eff_system else None),
            system_prompt_length=(len(eff_system) if eff_system else 0),
            prompt_preview=prompt[:200],
            prompt_length=len(prompt),
            conv_history_length=len(conversation_history or []),
            latency_ms=t.ms,
            **(trace_extra or {}),
            input_tokens=None,
            output_tokens=None,
            total_tokens=None,
        )
        return result

    def json_response(
        self,
        prompt: str,
        schema: Optional[Type[BaseModel]] = None,
        system_prompt: Optional[str] = None,
        conversation_history: MessageList = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        temperature: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        tracer: Optional[Tracer] = None,
        image: Optional[Any] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        txt = self.response(
            prompt=f"{prompt}\n\nReturn only a strict JSON object.",
            system_prompt=system_prompt,
            conversation_history=conversation_history,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            max_output_tokens=max_output_tokens,
            tracer=tracer,
            image=image,
            **kwargs,
        )
        return self._parse_json_response(txt or "{}", schema=schema)

    def response_stream(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        conversation_history: MessageList = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        temperature: Optional[float] = None,
        presence_penalty: Optional[float] = None,
        frequency_penalty: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        tracer: Optional[Tracer] = None,
        image: Optional[Any] = None,
        *args: Any,
        **kwargs: Any,
    ) -> Generator[str, None, None]:
        if not self.model:
            raise ValueError("model is required (pass in clientDetails.params.model).")

        eff_system = self._effective_param("system_prompt", system_prompt)
        eff_top_k = self._effective_param("top_k", top_k)
        eff_top_p = self._effective_param("top_p", top_p)
        eff_temp = self._effective_param("temperature", temperature)
        eff_pres = self._effective_param("presence_penalty", presence_penalty)
        eff_freq = self._effective_param("frequency_penalty", frequency_penalty)
        eff_max = self._effective_param("max_output_tokens", max_output_tokens)

        _hf_guardrails(eff_top_k, eff_pres, eff_freq)

        conversation_history = self.trim_conversation_history(conversation_history)

        tr = _lazy_import_transformers()
        TextIteratorStreamer = tr.TextIteratorStreamer
        local_dir = self._resolve_local_dir(self.model)

        stream_gen_kwargs: Dict[str, Any] = {}
        if eff_max is not None:
            stream_gen_kwargs["max_new_tokens"] = eff_max
        if eff_temp is not None:
            stream_gen_kwargs["temperature"] = eff_temp
        if eff_top_p is not None:
            stream_gen_kwargs["top_p"] = eff_top_p
        if eff_top_k is not None:
            stream_gen_kwargs["top_k"] = eff_top_k
        if any(k in stream_gen_kwargs for k in ("temperature", "top_p", "top_k")):
            stream_gen_kwargs["do_sample"] = True

        trace_extra = kwargs.pop("_trace", None)

        # --- Vision-language model path ---
        if image is not None and self._is_vision_model(local_dir):
            pil_image = self._prepare_image(image)
            vm = self._get_vision_model(self.model)
            processor, mdl = vm["processor"], vm["model"]

            text_prompt = ""
            if eff_system:
                text_prompt += f"{eff_system}\n\n"
            text_prompt += f"USER: <image>\n{prompt}\nASSISTANT:"

            inputs = processor(text=text_prompt, images=pil_image, return_tensors="pt")
            streamer = TextIteratorStreamer(processor.tokenizer, skip_prompt=True, skip_special_tokens=True)
            gen_kwargs = {**inputs, "streamer": streamer, **stream_gen_kwargs}

            def _gen():
                mdl.generate(**gen_kwargs)

            with self._timed() as t:
                th = threading.Thread(target=_gen, daemon=True)
                th.start()
                for piece in streamer:
                    yield piece
                th.join()

            self._trace(
                tracer,
                operation="response_stream",
                model=self.model,
                params={k: v for k, v in {"temperature": eff_temp, "top_p": eff_top_p, "top_k": eff_top_k, "max_output_tokens": eff_max}.items() if v is not None},
                system_prompt_preview=(eff_system[:200] if eff_system else None),
                system_prompt_length=(len(eff_system) if eff_system else 0),
                prompt_preview=prompt[:200],
                prompt_length=len(prompt),
                conv_history_length=len(conversation_history or []),
                image_provided=True,
                latency_ms=t.ms,
                **(trace_extra or {}),
            )
            return

        # --- Text-only path (original) ---
        full_prompt = ""
        if eff_system:
            full_prompt += f"{eff_system}\n\n"
        if conversation_history:
            for m in conversation_history:
                full_prompt += f"{m.get('role','user')}: {m.get('content','')}\n"
        full_prompt += f"user: {prompt}\nassistant:"

        AutoTokenizer = tr.AutoTokenizer
        AutoModelForCausalLM = tr.AutoModelForCausalLM

        tok = AutoTokenizer.from_pretrained(local_dir)
        mdl = AutoModelForCausalLM.from_pretrained(local_dir)
        streamer = TextIteratorStreamer(tok, skip_prompt=True, skip_special_tokens=True)

        inputs = tok(full_prompt, return_tensors="pt")
        gen_kwargs = {"input_ids": inputs.input_ids, "streamer": streamer, **stream_gen_kwargs}

        def _gen():
            mdl.generate(**gen_kwargs)

        with self._timed() as t:
            th = threading.Thread(target=_gen, daemon=True)
            th.start()
            for piece in streamer:
                yield piece
            th.join()

        self._trace(
            tracer,
            operation="response_stream",
            model=self.model,
            params={k: v for k, v in {"temperature": eff_temp, "top_p": eff_top_p, "top_k": eff_top_k, "max_output_tokens": eff_max}.items() if v is not None},
            system_prompt_preview=(eff_system[:200] if eff_system else None),
            system_prompt_length=(len(eff_system) if eff_system else 0),
            prompt_preview=prompt[:200],
            prompt_length=len(prompt),
            conv_history_length=len(conversation_history or []),
            latency_ms=t.ms,
            **(trace_extra or {}),
        )
