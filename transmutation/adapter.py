from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import torch
from torch import nn


@dataclass
class ThoughtAdapter:
    """Applies a transmuted low-rank update at a chosen hook point.

    This is intentionally lightweight: it adds a residual projection
    using the aggregated weight/bias (Eq. 8, 24) from the transmuting
    prompts paper.
    """

    bias_delta: torch.Tensor
    weight_delta: torch.Tensor

    def __call__(self, hidden: torch.Tensor) -> torch.Tensor:
        # hidden: [*, hidden_size]
        return hidden + hidden @ self.weight_delta.T + self.bias_delta


def register_thought_hook(
    module: nn.Module,
    adapter: ThoughtAdapter,
    hook_selector: Callable[[nn.Module], nn.Module],
) -> torch.utils.hooks.RemovableHandle:
    """Register a forward hook that applies the adapter to a target submodule.

    Args:
        module: root module (e.g., model).
        adapter: computed ThoughtAdapter.
        hook_selector: function that returns the submodule to patch.

    Returns:
        A RemovableHandle so the caller can remove the hook.
    """

    target = hook_selector(module)

    def _hook(_mod, _inp, out):
        return adapter(out)

    return target.register_forward_hook(_hook)
