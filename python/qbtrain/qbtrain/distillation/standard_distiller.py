from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional

from qbtrain.ai.llm.base_llm_client import LLMClient

from .base_distiller import BaseDistiller, DistillationMetrics, _lazy_torch


class StandardDistiller(BaseDistiller):
    """Standard Knowledge Distillation (Hinton et al. 2015).

    The teacher LLM generates completions for each training prompt.  The student
    is trained with next-token prediction on the teacher-generated text using
    cross-entropy loss.

    Because the teacher is accessed through the ``LLMClient`` interface (no
    direct logit access), this is *response-based* distillation: the student
    learns to reproduce the teacher's textual output.
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
        temperature: float = 1.0,
        max_seq_length: int = 256,
    ) -> None:
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.temperature = temperature
        self.max_seq_length = max_seq_length
        super().__init__(teacher, student_model_name, output_dir, device=device)
        self._setup_optimizer()

    def _init_method_state(self, **kwargs: Any) -> None:
        self.learning_rate = kwargs.get("learning_rate", 5e-5)
        self.batch_size = kwargs.get("batch_size", 4)
        self.temperature = kwargs.get("temperature", 1.0)
        self.max_seq_length = kwargs.get("max_seq_length", 256)
        self._setup_optimizer()

    def _setup_optimizer(self) -> None:
        torch = _lazy_torch()
        self._optimizer = torch.optim.AdamW(self._model.parameters(), lr=self.learning_rate)

    def distill_step(self, train_texts: List[str], **kwargs: Any) -> DistillationMetrics:
        torch = _lazy_torch()
        t0 = time.perf_counter()

        # Get teacher completions
        completions = self._get_teacher_completions(train_texts)

        # Tokenize
        input_ids, labels = self._tokenize_pairs(train_texts, completions, self.max_seq_length)

        # Train in mini-batches
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

            if self.temperature != 1.0:
                # Scale logits by temperature for softer distribution
                logits = outputs.logits / self.temperature
                log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
                # Re-compute loss with temperature-scaled logits
                shift_log_probs = log_probs[:, :-1, :].contiguous()
                shift_labels = batch_labels[:, 1:].contiguous()
                loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fn(
                    shift_log_probs.view(-1, shift_log_probs.size(-1)),
                    shift_labels.view(-1),
                ) * (self.temperature ** 2)

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
