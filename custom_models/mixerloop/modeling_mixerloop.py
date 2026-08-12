# -*- coding: utf-8 -*-

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, Any, Optional, Tuple, Union

import torch
import torch.nn as nn
from fla.layers import GatedDeltaNet
from fla.models.gated_deltanet.modeling_gated_deltanet import GatedDeltaNetPreTrainedModel
from fla.modules import FusedCrossEntropyLoss, FusedLinearCrossEntropyLoss, RMSNorm
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
from transformers.utils import logging
from transformers.utils.deprecation import deprecate_kwarg

from .configuration_mixerloop import MixerLoopConfig
from .layers import MixerLoopBlock

if TYPE_CHECKING:
    from transformers.processing_utils import Unpack

logger = logging.get_logger(__name__)


class MixerLoopPreTrainedModel(GatedDeltaNetPreTrainedModel):

    config_class = MixerLoopConfig
    base_model_prefix = 'model'
    supports_gradient_checkpointing = True
    _no_split_modules = ['MixerLoopBlock']
    _supports_cache_class = False

    def _init_weights(self, module: nn.Module):
        super()._init_weights(module)
        if isinstance(module, GatedDeltaNet) and next(module.parameters()).device.type != 'meta':
            scale = math.sqrt(2 * self.config.num_hidden_layers * self.config.loop_count)
            nn.init.normal_(
                module.o_proj.weight,
                mean=0.0,
                std=self.config.initializer_range / scale,
            )


class MixerLoopModel(MixerLoopPreTrainedModel):

    def __init__(self, config: MixerLoopConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.loop_count = config.loop_count

        norm_cls = RMSNorm if config.fuse_norm else nn.RMSNorm
        self.embeddings = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            MixerLoopBlock(config, i) for i in range(config.num_hidden_layers)
        )
        self.norm = norm_cls(config.hidden_size, eps=config.norm_eps)

        if config.use_residual:
            self.residual_weight = nn.Parameter(
                torch.zeros(config.loop_count, config.hidden_size)
            )
        else:
            self.residual_weight = None

        self.gradient_checkpointing = False
        self.post_init()

    def reset_parameters(self):
        if self.residual_weight is not None:
            nn.init.zeros_(self.residual_weight)

    def get_input_embeddings(self):
        return self.embeddings

    def set_input_embeddings(self, value):
        self.embeddings = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs: Unpack[Any],
    ) -> Union[Tuple, BaseModelOutputWithPast]:
        if output_attentions:
            warnings.warn('MixerLoopModel does not support output_attentions; ignoring.')
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError('Specify either input_ids or inputs_embeds, not both.')
        if input_ids is None and inputs_embeds is None:
            raise ValueError('Must specify input_ids or inputs_embeds.')

        hidden_states = self.embeddings(input_ids) if inputs_embeds is None else inputs_embeds

        all_hidden_states = () if output_hidden_states else None

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)
            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    layer.__call__,
                    hidden_states,
                    self.residual_weight,
                    attention_mask,
                )
            else:
                hidden_states = layer(
                    hidden_states,
                    residual_weight=self.residual_weight,
                    attention_mask=attention_mask,
                    **kwargs,
                )

        hidden_states = self.norm(hidden_states)
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, None, all_hidden_states] if v is not None)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=None,
            hidden_states=all_hidden_states,
            attentions=None,
        )


class MixerLoopForCausalLM(MixerLoopPreTrainedModel, GenerationMixin):

    _tied_weights_keys = ['lm_head.weight']

    def __init__(self, config: MixerLoopConfig):
        super().__init__(config)
        self.model = MixerLoopModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.criterion = None
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embeddings

    def set_input_embeddings(self, value):
        self.model.embeddings = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    @deprecate_kwarg('num_logits_to_keep', version='4.50', new_name='logits_to_keep')
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        logits_to_keep: Optional[int] = 0,
        **kwargs: Unpack[Any],
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            **kwargs,
        )

        hidden_states = outputs[0]
        fuse_linear_and_cross_entropy = self.config.fuse_linear_cross_entropy and self.training
        logits = None if fuse_linear_and_cross_entropy else self.lm_head(hidden_states[:, -logits_to_keep:])

        loss = None
        if labels is not None:
            if getattr(self, 'criterion', None) is None:
                if fuse_linear_and_cross_entropy:
                    criterion = FusedLinearCrossEntropyLoss()
                elif self.config.fuse_cross_entropy:
                    criterion = FusedCrossEntropyLoss(inplace_backward=True)
                else:
                    criterion = nn.CrossEntropyLoss()
            else:
                criterion = self.criterion
            labels = labels.to(hidden_states.device)
            labels = torch.cat((labels[..., 1:], torch.full_like(labels[:, :1], criterion.ignore_index)), 1)
            if fuse_linear_and_cross_entropy:
                loss = criterion(hidden_states, labels, self.lm_head.weight, self.lm_head.bias)
            else:
                loss = criterion(logits.view(labels.numel(), -1), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=None,
            hidden_states=outputs.hidden_states,
            attentions=None,
        )
