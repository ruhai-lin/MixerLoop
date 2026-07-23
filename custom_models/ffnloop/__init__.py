from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_ffnloop import FFNLoopConfig
from .modeling_ffnloop import FFNLoopForCausalLM, FFNLoopModel

__all__ = ['FFNLoopConfig', 'FFNLoopForCausalLM', 'FFNLoopModel']

AutoConfig.register('ffnloop', FFNLoopConfig)
AutoModel.register(FFNLoopConfig, FFNLoopModel)
AutoModelForCausalLM.register(FFNLoopConfig, FFNLoopForCausalLM)
