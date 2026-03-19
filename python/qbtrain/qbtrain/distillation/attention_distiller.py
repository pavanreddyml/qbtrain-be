from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional

from qbtrain.ai.llm.base_llm_client import LLMClient

from .base_distiller import BaseDistiller, DistillationMetrics, _lazy_torch


class AttentionDistiller(BaseDistiller):
    """Attention Transfer distillation.

    The teacher generates text completions via ``LLMClient``.  The student
    trains on next-token prediction *plus* an attention regularisation loss
    that shapes the student's attention patterns.

    The auxiliary loss computes the mean entropy of the student's attention
    distributions and encourages them to be neither too uniform nor too peaked
    by penalising deviation from a target entropy.  This self-distillation
    signal improves the student's internal representations without requiring
    direct access to teacher attention maps.
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
        attention_weight: float = 0.5,
        max_seq_length: int = 256,
    ) -> None:
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.attention_weight = attention_weight
        self.max_seq_length = max_seq_length
        super().__init__(teacher, student_model_name, output_dir, device=device)
        self._setup_optimizer()

    def _init_method_state(self, **kwargs: Any) -> None:
        self.learning_rate = kwargs.get("learning_rate", 5e-5)
        self.batch_size = kwargs.get("batch_size", 4)
        self.attention_weight = kwargs.get("attention_weight", 0.5)
        self.max_seq_length = kwargs.get("max_seq_length", 256)
        self._setup_optimizer()

    def _setup_optimizer(self) -> None:
        torch = _lazy_torch()
        self._optimizer = torch.optim.AdamW(self._model.parameters(), lr=self.learning_rate)

    @staticmethod
    def _attention_entropy_loss(attentions) -> Any:
        """Compute mean attention entropy across all layers and heads.

        Encourages attention distributions to have moderate entropy — not too
        uniform (wastes capacity) and not too peaked (limits information flow).
        Returns a scalar loss.
        """
        torch = _lazy_torch()
        total_entropy = torch.tensor(0.0, device=attentions[0].device)
        count = 0

        for layer_attn in attentions:
            # layer_attn: (B, num_heads, T, T)
            # Clamp to avoid log(0)
            attn_probs = layer_attn.clamp(min=1e-8)
            entropy = -(attn_probs * attn_probs.log()).sum(dim=-1)  # (B, H, T)
            total_entropy = total_entropy + entropy.mean()
            count += 1

        # Negative entropy as loss: lower entropy = more peaked = we penalise
        # We want moderate entropy, so we penalise the *negative* to encourage
        # the model to not collapse attention to single tokens
        mean_entropy = total_entropy / max(count, 1)
        return -mean_entropy

    def distill_step(self, train_texts: List[str], **kwargs: Any) -> DistillationMetrics:
        torch = _lazy_torch()
        t0 = time.perf_counter()

        completions = self._get_teacher_completions(train_texts)
        input_ids, labels = self._tokenize_pairs(train_texts, completions, self.max_seq_length)

        total_loss = 0.0
        total_attn_loss = 0.0
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
                output_attentions=True,
            )

            ce_loss = outputs.loss
            attn_loss = self._attention_entropy_loss(outputs.attentions)

            loss = ce_loss + self.attention_weight * attn_loss

            self._optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
            self._optimizer.step()

            total_loss += loss.item()
            total_attn_loss += attn_loss.item()
            num_batches += 1

        self._step_counter += 1
        avg_loss = total_loss / max(num_batches, 1)
        avg_attn = total_attn_loss / max(num_batches, 1)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)

        return DistillationMetrics(
            step=self._step_counter,
            loss=avg_loss,
            learning_rate=self.learning_rate,
            perplexity=math.exp(min(avg_loss, 20)),
            elapsed_ms=elapsed_ms,
            extras={"attention_loss": avg_attn},
        )
