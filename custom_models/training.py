"""Training components that reproduce the optimizer contract of LT2.c/LT3.c."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from torchtitan.components.optimizer import OptimizersContainer


class ParameterGroupedOptimizersContainer(OptimizersContainer):
    """AdamW container with weight decay restricted to matrix parameters."""

    def __init__(
        self,
        model_parts: list[nn.Module],
        optimizer_cls: type[torch.optim.Optimizer],
        optimizer_kwargs: dict[str, Any],
    ) -> None:
        all_params: list[nn.Parameter] = []
        self.optimizers = []
        self.model_parts = model_parts
        weight_decay = float(optimizer_kwargs["weight_decay"])
        common_kwargs = {**optimizer_kwargs, "weight_decay": 0.0}

        for model in model_parts:
            decay: list[nn.Parameter] = []
            no_decay: list[nn.Parameter] = []
            for name, parameter in model.named_parameters():
                if not parameter.requires_grad:
                    continue
                if parameter.dim() < 2 or name.endswith("A_log") or name.endswith("dt_bias"):
                    no_decay.append(parameter)
                else:
                    decay.append(parameter)
            groups = [
                {"params": decay, "weight_decay": weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ]
            self.optimizers.append(optimizer_cls(groups, **common_kwargs))
            all_params.extend(decay)
            all_params.extend(no_decay)

        self._validate_length(len(model_parts))
        self._post_init(all_params, optimizer_kwargs)


def build_reproduction_optimizers(model_parts, job_config, ft_manager):
    """Build fused AdamW with the exact decay grouping used by legacy runs."""

    if job_config.optimizer.name != "AdamW":
        raise ValueError("MixerLoop reproduction recipes require AdamW")
    if job_config.optimizer.early_step_in_backward:
        raise NotImplementedError("Optimizer-in-backward is not supported by the reproduction recipe")
    if ft_manager.enabled:
        raise NotImplementedError("TorchFT is not supported by the reproduction optimizer")

    implementation = job_config.optimizer.implementation
    if implementation not in {"fused", "foreach", "for-loop"}:
        raise ValueError(f"Unknown optimizer implementation: {implementation}")
    kwargs = {
        "lr": job_config.optimizer.lr,
        "betas": (job_config.optimizer.beta1, job_config.optimizer.beta2),
        "eps": job_config.optimizer.eps,
        "weight_decay": job_config.optimizer.weight_decay,
        "fused": implementation == "fused",
        "foreach": implementation == "foreach",
    }
    return ParameterGroupedOptimizersContainer(model_parts, torch.optim.AdamW, kwargs)
