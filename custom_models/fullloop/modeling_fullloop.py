"""Pure-GDN FullLoop: https://github.com/chili-lab/LT2 (1ac9ed5)."""

import warnings

import torch.nn as nn
from fla.models.gated_deltanet.modeling_gated_deltanet import GatedDeltaNetPreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPast

from custom_models.mixerloop.layers import MixerLoopBlock
from custom_models.mixerloop.modeling_mixerloop import (
    MixerLoopForCausalLM,
    MixerLoopModel,
)

from .configuration_fullloop import FullLoopConfig


def _init_full_loop_weights(self, module):
    if isinstance(module, (nn.Linear, nn.Embedding)):
        # LT2 init_std_factor="disabled": no depth/loop scaling of projections.
        std = self.config.initializer_range
        nn.init.trunc_normal_(module.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv1d):
        module.reset_parameters()
    else:
        # Includes GDN A_log/dt_bias and all normalization weights.
        GatedDeltaNetPreTrainedModel._init_weights(self, module)
    if isinstance(module, FullLoopModel):
        nn.init.zeros_(module.residual_weight)


class FullLoopBlock(MixerLoopBlock):
    """One native GDN residual followed by one FFN residual; no inner loop."""

    def forward(self, hidden_states, attention_mask=None, **kwargs):
        attn_out, _, _ = self.mixer(
            self.attn_norm(hidden_states),
            attention_mask=attention_mask,
            use_cache=False,
            **kwargs,
        )
        hidden_states = hidden_states + attn_out
        return hidden_states + self.ffn(self.ffn_norm(hidden_states))


class FullLoopModel(MixerLoopModel):
    config_class = FullLoopConfig
    block_class = FullLoopBlock
    _no_split_modules = ['FullLoopBlock']
    _init_weights = _init_full_loop_weights

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        use_cache=None,
        past_key_values=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        **kwargs,
    ):
        if output_attentions:
            warnings.warn('FullLoopModel does not support output_attentions; ignoring.')
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError('Specify exactly one of input_ids or inputs_embeds.')

        h = self.embeddings(input_ids) if inputs_embeds is None else inputs_embeds
        all_hidden_states = () if output_hidden_states else None

        for loop_idx in range(self.loop_count):
            h_input = h
            for layer in self.layers:
                if output_hidden_states:
                    all_hidden_states += (h,)
                if self.gradient_checkpointing and self.training:
                    h = self._gradient_checkpointing_func(layer.__call__, h, attention_mask, **kwargs)
                else:
                    h = layer(h, attention_mask=attention_mask, **kwargs)
            h = h + self.residual_weight[loop_idx] * h_input

        h = self.norm(h)
        if output_hidden_states:
            all_hidden_states += (h,)
        if not return_dict:
            return tuple(value for value in (h, None, all_hidden_states) if value is not None)
        return BaseModelOutputWithPast(last_hidden_state=h, hidden_states=all_hidden_states)


class FullLoopForCausalLM(MixerLoopForCausalLM):
    """Reuse the same HF interface, tied embeddings and causal loss as MixerLoop."""

    config_class = FullLoopConfig
    model_class = FullLoopModel
    _no_split_modules = ['FullLoopBlock']
    _init_weights = _init_full_loop_weights
