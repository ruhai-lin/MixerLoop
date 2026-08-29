from __future__ import annotations

import torch
import torch.nn as nn

from custom_models.training import ParameterGroupedOptimizersContainer


class ContractModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.matrix = nn.Parameter(torch.zeros(4, 4))
        self.norm = nn.Parameter(torch.ones(4))
        self.A_log = nn.Parameter(torch.zeros(2, 2))
        self.dt_bias = nn.Parameter(torch.zeros(2, 2))
        self.residual_weight = nn.Parameter(torch.zeros(4, 4))


def test_lt3_optimizer_parameter_groups():
    model = ContractModel()
    optimizers = ParameterGroupedOptimizersContainer(
        [model],
        torch.optim.AdamW,
        {
            "lr": 5e-4,
            "betas": (0.9, 0.95),
            "eps": 1e-8,
            "weight_decay": 0.1,
            "fused": False,
            "foreach": False,
        },
    )
    groups = optimizers.optimizers[0].param_groups
    decay = {id(parameter) for parameter in groups[0]['params']}
    no_decay = {id(parameter) for parameter in groups[1]['params']}

    assert groups[0]['weight_decay'] == 0.1
    assert groups[1]['weight_decay'] == 0.0
    assert id(model.residual_weight) in decay

    for name, parameter in model.named_parameters():
        no_decay_parameter = parameter.dim() < 2 or name.endswith(('A_log', 'dt_bias'))
        expected = no_decay if no_decay_parameter else decay
        assert id(parameter) in expected, name
