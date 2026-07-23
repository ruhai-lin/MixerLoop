from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

from .configuration_fullloop import FullLoopConfig
from .modeling_fullloop import FullLoopForCausalLM, FullLoopModel

__all__ = ['FullLoopConfig', 'FullLoopForCausalLM', 'FullLoopModel']

AutoConfig.register('fullloop', FullLoopConfig)
AutoModel.register(FullLoopConfig, FullLoopModel)
AutoModelForCausalLM.register(FullLoopConfig, FullLoopForCausalLM)
