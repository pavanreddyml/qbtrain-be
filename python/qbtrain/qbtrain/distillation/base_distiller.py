from __future__ import annotations

import importlib
import math
import shutil
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional

from qbtrain.ai.llm.base_llm_client import LLMClient
from qbtrain.exceptions.exceptions import StudentModelTooLargeError

MAX_STUDENT_PARAMS = 100_000_000


def _lazy_torch():
    return importlib.import_module("torch")


def _lazy_transformers():
    return importlib.import_module("transformers")


def _lazy_hf_hub():
    return importlib.import_module("huggingface_hub")


@dataclass
class DistillationMetrics:
    step: int
    loss: float
    learning_rate: float
    perplexity: float
    elapsed_ms: int
    extras: Dict[str, float] = field(default_factory=dict)


class BaseDistiller(ABC):
    """Abstract base for all distillation methods.

    The student model is downloaded once and *copied* into ``output_dir`` so the
    original cached weights are never modified.  Distillation is step-based:
    after any step you can call ``generate()`` to evaluate quality, or switch to
    a different distiller via ``from_existing()``.
    """

    def __init__(
        self,
        teacher: LLMClient,
        student_model_name: str,
        output_dir: str,
        *,
        device: str = "cpu",
    ) -> None:
        self.teacher = teacher
        self.student_model_name = student_model_name
        self.output_dir = Path(output_dir)
        self.device = device
        self._step_counter = 0

        self.output_dir.mkdir(parents=True, exist_ok=True)

        num_params = self.validate_student()
        self._num_params = num_params

        self._model, self._tokenizer = self._download_and_copy()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate_student(self) -> int:
        """Check the student model has at most 100M parameters.

        Uses the HuggingFace Hub API so the full model does not need to be
        downloaded just to check the size.

        Returns the parameter count.  Raises ``StudentModelTooLargeError`` if
        the model is too large.
        """
        hf_hub = _lazy_hf_hub()
        try:
            info = hf_hub.model_info(self.student_model_name)
        except Exception as exc:
            raise ValueError(
                f"Could not fetch model info for '{self.student_model_name}' from HuggingFace Hub."
            ) from exc

        num_params = getattr(info, "safetensors", None)
        if num_params is not None:
            # safetensors metadata stores total parameter count
            num_params = num_params.get("total", None) if isinstance(num_params, dict) else None

        if num_params is None:
            # Fallback: load just the config to estimate from architecture
            transformers = _lazy_transformers()
            config = transformers.AutoConfig.from_pretrained(self.student_model_name)
            # Rough estimate from config vocab/hidden sizes
            num_params = _estimate_params_from_config(config)

        if num_params > MAX_STUDENT_PARAMS:
            raise StudentModelTooLargeError(
                f"Student model '{self.student_model_name}' has ~{num_params:,} parameters, "
                f"exceeding the {MAX_STUDENT_PARAMS:,} limit."
            )
        return num_params

    # ------------------------------------------------------------------
    # Download & copy
    # ------------------------------------------------------------------

    def _download_and_copy(self):
        """Download the student model to HF cache, then copy to output_dir."""
        transformers = _lazy_transformers()

        # Download to default HF cache (shared, never modified)
        model = transformers.AutoModelForCausalLM.from_pretrained(self.student_model_name)
        tokenizer = transformers.AutoTokenizer.from_pretrained(self.student_model_name)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # Save a working copy into output_dir
        copy_dir = self.output_dir / "student_working_copy"
        copy_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(str(copy_dir))
        tokenizer.save_pretrained(str(copy_dir))

        # Reload from the copy so all further changes only affect the copy
        model = transformers.AutoModelForCausalLM.from_pretrained(str(copy_dir))
        tokenizer = transformers.AutoTokenizer.from_pretrained(str(copy_dir))
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        torch = _lazy_torch()
        model = model.to(self.device)
        model.train()

        return model, tokenizer

    # ------------------------------------------------------------------
    # Generation (evaluate quality anytime)
    # ------------------------------------------------------------------

    def generate(self, prompt: str, *, max_new_tokens: int = 100) -> str:
        """Generate text with the current student model state."""
        torch = _lazy_torch()
        self._model.eval()
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self._tokenizer.pad_token_id,
            )
        self._model.train()
        generated = output_ids[0][inputs["input_ids"].shape[1]:]
        return self._tokenizer.decode(generated, skip_special_tokens=True)

    # ------------------------------------------------------------------
    # Core distillation interface
    # ------------------------------------------------------------------

    @abstractmethod
    def distill_step(self, train_texts: List[str], **kwargs: Any) -> DistillationMetrics:
        """Run a single distillation step. Returns metrics for that step."""
        raise NotImplementedError

    def distill_stream(
        self,
        train_texts: List[str],
        steps: int = 10,
        **kwargs: Any,
    ) -> Generator[DistillationMetrics, None, None]:
        """Run ``steps`` distillation steps, yielding metrics after each one.

        The caller can iterate this generator and display metrics in a UI, stop
        early, or switch distillers between yields.
        """
        for _ in range(steps):
            yield self.distill_step(train_texts, **kwargs)

    # ------------------------------------------------------------------
    # Save / accessors
    # ------------------------------------------------------------------

    def save(self, path: Optional[str] = None) -> str:
        """Save the current student model and tokenizer.  Returns the save path."""
        save_dir = Path(path) if path else self.output_dir / "distilled_model"
        save_dir.mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(str(save_dir))
        self._tokenizer.save_pretrained(str(save_dir))
        return str(save_dir)

    def get_student_model(self):
        """Return the raw student model (for switching distillers)."""
        return self._model

    def get_tokenizer(self):
        """Return the tokenizer."""
        return self._tokenizer

    # ------------------------------------------------------------------
    # Construct from an already-loaded student (switch methods mid-way)
    # ------------------------------------------------------------------

    @classmethod
    def from_existing(
        cls,
        teacher: LLMClient,
        model: Any,
        tokenizer: Any,
        output_dir: str,
        *,
        device: str = "cpu",
        **kwargs: Any,
    ):
        """Create a distiller re-using an already-loaded student model.

        This lets you switch distillation methods without re-downloading or
        re-initialising the student.
        """
        instance = object.__new__(cls)
        instance.teacher = teacher
        instance.student_model_name = getattr(model, "name_or_path", "unknown")
        instance.output_dir = Path(output_dir)
        instance.device = device
        instance._step_counter = 0
        instance._model = model.to(device)
        instance._model.train()
        instance._tokenizer = tokenizer
        if instance._tokenizer.pad_token is None:
            instance._tokenizer.pad_token = instance._tokenizer.eos_token
        instance._num_params = sum(p.numel() for p in model.parameters())
        instance.output_dir.mkdir(parents=True, exist_ok=True)
        instance._init_method_state(**kwargs)
        return instance

    def _init_method_state(self, **kwargs: Any) -> None:
        """Override in subclasses to initialise method-specific state in from_existing()."""
        pass

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_teacher_completions(self, prompts: List[str]) -> List[str]:
        """Ask the teacher LLM for completions of each prompt."""
        completions = []
        for prompt in prompts:
            text = self.teacher.response(prompt)
            completions.append(text)
        return completions

    def _tokenize_pairs(self, prompts: List[str], completions: List[str], max_length: int = 256):
        """Tokenize prompt+completion pairs for causal LM training."""
        torch = _lazy_torch()
        all_input_ids = []
        all_labels = []

        for prompt, completion in zip(prompts, completions):
            full_text = prompt + completion
            encoded = self._tokenizer(
                full_text,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            input_ids = encoded["input_ids"].squeeze(0)

            # Labels: mask the prompt portion with -100
            prompt_ids = self._tokenizer(
                prompt,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )["input_ids"].squeeze(0)
            labels = input_ids.clone()
            labels[: len(prompt_ids)] = -100

            all_input_ids.append(input_ids)
            all_labels.append(labels)

        # Pad to same length
        input_ids_padded = torch.nn.utils.rnn.pad_sequence(
            all_input_ids, batch_first=True, padding_value=self._tokenizer.pad_token_id
        )
        labels_padded = torch.nn.utils.rnn.pad_sequence(
            all_labels, batch_first=True, padding_value=-100
        )

        return input_ids_padded.to(self.device), labels_padded.to(self.device)


def _estimate_params_from_config(config) -> int:
    """Rough parameter estimate from a transformers config."""
    vocab = getattr(config, "vocab_size", 50257)
    hidden = getattr(config, "hidden_size", getattr(config, "n_embd", 768))
    layers = getattr(config, "num_hidden_layers", getattr(config, "n_layer", 12))
    intermediate = getattr(config, "intermediate_size", hidden * 4)

    embed_params = vocab * hidden
    # Each transformer layer: attention (4 * h*h) + FFN (2 * h * intermediate) + layernorms
    layer_params = 4 * hidden * hidden + 2 * hidden * intermediate + 4 * hidden
    total = embed_params + layers * layer_params
    return total
