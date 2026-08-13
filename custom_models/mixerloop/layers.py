# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from fla.layers import GatedDeltaNet
from fla.modules import GatedMLP, RMSNorm

from .configuration_mixerloop import MixerLoopConfig


class MixerLoopBlock(nn.Module):
    """GDN mixer repeated loop_count times, then one GatedMLP."""

    def __init__(self, config: MixerLoopConfig, layer_idx: int):
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
        attention_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        h = hidden_states
        for _ in range(self.loop_count):
            attn_out, _, _ = self.mixer(
                self.attn_norm(h),
                attention_mask=attention_mask,
                **kwargs,
            )
            h = h + attn_out
        return h + self.ffn(self.ffn_norm(h))
