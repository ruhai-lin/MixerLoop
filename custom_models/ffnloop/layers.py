# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from fla.layers import GatedDeltaNet
from fla.modules import GatedMLP, RMSNorm

from .configuration_ffnloop import FFNLoopConfig


class FFNLoopBlock(nn.Module):
    """One GDN mixer pass followed by loop_count shared GatedMLP passes."""

    def __init__(self, config: FFNLoopConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.loop_count = config.loop_count

        norm_cls = RMSNorm if config.fuse_norm else nn.RMSNorm
        self.attn_norm = norm_cls(config.hidden_size, eps=config.norm_eps)
        self.mixer = GatedDeltaNet(
            hidden_size=config.hidden_size,
            expand_v=config.expand_v,
            head_dim=config.head_dim,
            num_heads=config.num_heads,
            mode=config.attn_mode,
            use_gate=True,
            use_short_conv=True,
            conv_size=config.conv_size,
            norm_eps=config.norm_eps,
            layer_idx=layer_idx,
        )
        self.ffn_norm = norm_cls(config.hidden_size, eps=config.norm_eps)
        self.ffn = GatedMLP(
            hidden_size=config.hidden_size,
            hidden_ratio=config.hidden_ratio,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            fuse_swiglu=config.fuse_swiglu,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual_weight: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        attn_out, _, _ = self.mixer(
            self.attn_norm(hidden_states),
            attention_mask=attention_mask,
            **kwargs,
        )
        h = hidden_states + attn_out
        for loop_idx in range(self.loop_count):
            h_input = h
            h = h + self.ffn(self.ffn_norm(h))
            if residual_weight is not None:
                h = h + residual_weight[loop_idx].view(1, 1, -1) * h_input
        return h
