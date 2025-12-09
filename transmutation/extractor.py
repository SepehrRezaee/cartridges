from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Optional

import torch
from torch import nn
from transformers import PreTrainedTokenizerBase

from cartridges.datasets import TrainDataset, DatasetElement


@dataclass
class TokenPatch:
    """Container for a single token-level patch.

    References:
        Eq. 3, 4 in "Transmuting Prompts into Weights".
    """

    delta: torch.Tensor  # A(C, t) - A(empty, t)
    baseline: torch.Tensor  # A(empty, t)
    element_idx: int
    token_idx: int


class TokenPatchExtractor:
    """Extract token-level activation deltas for a corpus.

    For each token t we compute A(C, t) with context C, A(empty, t)
    without context, the delta_t = A(C, t) - A(empty, t) (Eq. 3) and
    keep baseline activations for later rank-one updates (Eq. 4).
    """

    def __init__(
        self,
        model: nn.Module,
        tokenizer: PreTrainedTokenizerBase,
        device: str = "cuda",
    ) -> None:
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        # ensure we expose hidden states for activations
        self.model.eval()

    def _forward_hidden(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run a forward pass and return last hidden state."""
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                output_hidden_states=True,
            )
        hidden: torch.Tensor = outputs.hidden_states[-1]
        return hidden.squeeze(0)

    def extract(
        self,
        dataset: TrainDataset,
        context_strip_fn: Optional[Callable[[DatasetElement], torch.Tensor]] = None,
        max_batches: Optional[int] = None,
    ) -> List[TokenPatch]:
        """Compute TokenPatch objects over the dataset.

        Args:
            dataset: The synthetic corpus (from self-study) wrapped in a TrainDataset.
            context_strip_fn: Callable that returns a "no-context" input tensor.
                Defaults to reusing the trailing tokens of the element.
            max_batches: optional limit for quick debugging.
        """
        patches: List[TokenPatch] = []
        batches = dataset.batches[:max_batches] if max_batches else dataset.batches

        for batch in batches:
            for elem_idx in batch:
                element = dataset._get_element(elem_idx)
                patches.extend(
                    self._extract_from_element(
                        element,
                        element_idx=elem_idx,
                        context_strip_fn=context_strip_fn,
                    )
                )
        return patches

    def _default_strip(self, element: DatasetElement) -> torch.Tensor:
        """Fallback 'no-context' input: keep the assistant reply only.

        This is a conservative heuristic when an explicit context remover
        is not provided.
        """
        # take the last half of tokens as a crude approximation of reply-only
        cutoff = max(1, element.input_ids.numel() // 2)
        return element.input_ids[-cutoff:]

    def _extract_from_element(
        self,
        element: DatasetElement,
        element_idx: int,
        context_strip_fn: Optional[Callable[[DatasetElement], torch.Tensor]],
    ) -> Iterable[TokenPatch]:
        with_ids = element.input_ids.to(self.device)
        base_ids = (
            context_strip_fn(element).to(self.device)
            if context_strip_fn
            else self._default_strip(element).to(self.device)
        )

        # Align sequence lengths to compare token by token.
        act_with = self._forward_hidden(with_ids)
        act_base = self._forward_hidden(base_ids)
        seq_len = min(act_with.size(0), act_base.size(0))

        for t in range(seq_len):
            baseline = act_base[t]
            delta = act_with[t] - baseline
            yield TokenPatch(
                delta=delta.detach().cpu(),
                baseline=baseline.detach().cpu(),
                element_idx=element_idx,
                token_idx=t,
            )
