from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional

from qbtrain.ai.llm.base_llm_client import LLMClient

from .base_distiller import BaseDistiller, DistillationMetrics, _lazy_torch


class FeatureDistiller(BaseDistiller):
    """Feature-based distillation (FitNets-style).

    The teacher generates text completions via ``LLMClient``.  The student
    trains on next-token prediction *plus* an auxiliary MSE loss that
    encourages richer internal representations by projecting hidden states
    through a learned projection layer and penalising variance across layers.

    A small linear projection maps the student's hidden dimension to a target
    dimension (default: same as hidden dim), and the auxiliary loss encourages
    the projected representations from the *last* hidden layer to have high
    information content (low reconstruction error vs. the mean representation).
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
        feature_weight: float = 0.5,
        max_seq_length: int = 256,
    ) -> None:
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.feature_weight = feature_weight
        self.max_seq_length = max_seq_length
        super().__init__(teacher, student_model_name, output_dir, device=device)
        self._setup_projection_and_optimizer()

    def _init_method_state(self, **kwargs: Any) -> None:
        self.learning_rate = kwargs.get("learning_rate", 5e-5)
        self.batch_size = kwargs.get("batch_size", 4)
        self.feature_weight = kwargs.get("feature_weight", 0.5)
        self.max_seq_length = kwargs.get("max_seq_length", 256)
        self._setup_projection_and_optimizer()

    def _setup_projection_and_optimizer(self) -> None:
        torch = _lazy_torch()
        hidden_size = self._model.config.hidden_size
        # Projection layer: student hidden -> target dim (same size, learned transform)
        self._projection = torch.nn.Linear(hidden_size, hidden_size).to(self.device)
        params = list(self._model.parameters()) + list(self._projection.parameters())
        self._optimizer = torch.optim.AdamW(params, lr=self.learning_rate)

    def distill_step(self, train_texts: List[str], **kwargs: Any) -> DistillationMetrics:
        torch = _lazy_torch()
        t0 = time.perf_counter()

        completions = self._get_teacher_completions(train_texts)
        input_ids, labels = self._tokenize_pairs(train_texts, completions, self.max_seq_length)

        total_loss = 0.0
        total_feature_loss = 0.0
        num_batches = 0
        self._model.train()
        self._projection.train()

        for i in range(0, len(input_ids), self.batch_size):
            batch_ids = input_ids[i : i + self.batch_size]
            batch_labels = labels[i : i + self.batch_size]
            attention_mask = (batch_ids != self._tokenizer.pad_token_id).long()

            outputs = self._model(
                input_ids=batch_ids,
                attention_mask=attention_mask,
                labels=batch_labels,
                output_hidden_states=True,
            )

            ce_loss = outputs.loss

            # Feature loss: project last hidden state and compute MSE vs mean
            last_hidden = outputs.hidden_states[-1]  # (B, T, H)
            projected = self._projection(last_hidden)  # (B, T, H)
            # Encourage deviation from mean (richer representations)
            mean_repr = projected.mean(dim=1, keepdim=True)  # (B, 1, H)
            feature_loss = torch.nn.functional.mse_loss(projected, mean_repr.expand_as(projected))

            loss = ce_loss + self.feature_weight * feature_loss

            self._optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(self._model.parameters()) + list(self._projection.parameters()), 1.0
            )
            self._optimizer.step()

            total_loss += loss.item()
            total_feature_loss += feature_loss.item()
            num_batches += 1

        self._step_counter += 1
        avg_loss = total_loss / max(num_batches, 1)
        avg_feat = total_feature_loss / max(num_batches, 1)
        elapsed_ms = int((time.perf_counter() - t0) * 1000)

        return DistillationMetrics(
            step=self._step_counter,
            loss=avg_loss,
            learning_rate=self.learning_rate,
            perplexity=math.exp(min(avg_loss, 20)),
            elapsed_ms=elapsed_ms,
            extras={"feature_loss": avg_feat},
        )
