from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM
                         
from transformers.modeling_outputs import CausalLMOutputWithPast

from ..emollava_arch import EmoLlavaMetaModel, EmoLlavaMetaForCausalLM


class EmoLlavaConfig(LlamaConfig):
    model_type = 'emollava'


class EmoLlavaLlamaModel(EmoLlavaMetaModel, LlamaModel):
    config_class = EmoLlavaConfig
    
    def __init__(self, config: LlamaConfig):
        super(EmoLlavaLlamaModel, self).__init__(config)

        
class EmoLlavaLlamaForCausalLM(LlamaForCausalLM, EmoLlavaMetaForCausalLM):
    config_class = EmoLlavaConfig
    
    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = EmoLlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        self.post_init()
    
    def get_model(self):
        return self.model
    
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        org_landmark_features: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                org_landmark_features
            )

        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        org_landmark_features = kwargs.pop("org_landmark_features", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            _inputs['images'] = images
        if org_landmark_features is not None:
            _inputs['org_landmark_features'] = org_landmark_features
        return _inputs

AutoConfig.register("emollava", EmoLlavaConfig)
AutoModelForCausalLM.register(EmoLlavaConfig, EmoLlavaLlamaForCausalLM)
