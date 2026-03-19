from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional

from qbtrain.ai.llm.base_llm_client import LLMClient

from .base_distiller import BaseDistiller, DistillationMetrics, _lazy_torch


DEFAULT_TASK_TEMPLATE = (
    "Given the following input, produce a helpful response.\n\nInput: {input}\n\nResponse:"
)


class TaskDistiller(BaseDistiller):
    """Task-specific distillation.

    The teacher generates structured prompt-completion pairs via
    ``LLMClient.json_response()``.  For each training text, the teacher is
    asked (using ``task_prompt_template``) to produce a JSON object with
    ``"input"`` and ``"output"`` fields.  The student is then fine-tuned on
    these pairs with standard causal LM loss.

    This is useful when you want the student to learn a specific task
    (summarisation, Q&A, classification-as-text, etc.) rather than general
    language modelling.
    """

    def __init__(
        self,
        teacher: LLMClient,
        student_model_name: str,
        output_dir: str,
        *,
        device: str = "cpu",
        learning_rate: float = 5e-5,
        batch_size: int = 4,
        task_prompt_template: str = DEFAULT_TASK_TEMPLATE,
        max_seq_length: int = 256,
    ) -> None:
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.task_prompt_template = task_prompt_template
        self.max_seq_length = max_seq_length
        super().__init__(teacher, student_model_name, output_dir, device=device)
        self._setup_optimizer()

    def _init_method_state(self, **kwargs: Any) -> None:
        self.learning_rate = kwargs.get("learning_rate", 5e-5)
        self.batch_size = kwargs.get("batch_size", 4)
        self.task_prompt_template = kwargs.get("task_prompt_template", DEFAULT_TASK_TEMPLATE)
        self.max_seq_length = kwargs.get("max_seq_length", 256)
        self._setup_optimizer()

    def _setup_optimizer(self) -> None:
        torch = _lazy_torch()
        self._optimizer = torch.optim.AdamW(self._model.parameters(), lr=self.learning_rate)

    def _get_task_pairs(self, train_texts: List[str]) -> List[tuple[str, str]]:
        """Ask the teacher to generate structured input/output pairs.

        For each training text, we ask the teacher to return JSON with
        ``input`` and ``output`` fields.  Falls back to using the raw text as
        input and the teacher's plain response as output if JSON parsing fails.
        """
        pairs: List[tuple[str, str]] = []

        for text in train_texts:
            teacher_prompt = (
                f"For the following text, produce a JSON object with two fields: "
                f'"input" (a prompt derived from the text) and "output" (the ideal response).\n\n'
                f"Text: {text}\n\n"
                f"Return only valid JSON."
            )
            try:
                result = self.teacher.json_response(teacher_prompt)
                task_input = str(result.get("input", text))
                task_output = str(result.get("output", ""))
            except Exception:
                # Fallback: use raw text + teacher plain response
                task_input = text
                task_output = self.teacher.response(text)

            prompt = self.task_prompt_template.format(input=task_input)
            pairs.append((prompt, task_output))

        return pairs

    def distill_step(self, train_texts: List[str], **kwargs: Any) -> DistillationMetrics:
        torch = _lazy_torch()
        t0 = time.perf_counter()

        # Get task-specific pairs from teacher
        pairs = self._get_task_pairs(train_texts)
        prompts = [p for p, _ in pairs]
        completions = [c for _, c in pairs]

        input_ids, labels = self._tokenize_pairs(prompts, completions, self.max_seq_length)

        total_loss = 0.0
        num_batches = 0
        self._model.train()

        for i in range(0, len(input_ids), self.batch_size):
            batch_ids = input_ids[i : i + self.batch_size]
            batch_labels = labels[i : i + self.batch_size]
            attention_mask = (batch_ids != self._tokenizer.pad_token_id).long()

            outputs = self._model(
                input_ids=batch_ids,
                attention_mask=attention_mask,
                labels=batch_labels,
            )
            loss = outputs.loss

            self._optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
            self._optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        self._step_counter += 1
        avg_loss = total_loss / max(num_batches, 1)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)

        return DistillationMetrics(
            step=self._step_counter,
            loss=avg_loss,
            learning_rate=self.learning_rate,
            perplexity=math.exp(min(avg_loss, 20)),
            elapsed_ms=elapsed_ms,
        )
