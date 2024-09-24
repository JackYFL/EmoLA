import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaForCausalLM
                         
from transformers.modeling_outputs import CausalLMOutputWithPast
from torch.nn import CrossEntropyLoss

from typing import List, Optional, Tuple, Union

from ..llavaquery_arch import LlavaMetaForCausalLMQuery
from .llava_llama import LlavaLlamaModel, LlavaConfig
from ..deepquery import DeepQuery, IConDeepQuery


class LlavaLlamaForCausalLMDeepQuery(LlamaForCausalLM, LlavaMetaForCausalLMQuery):
    config_class = LlavaConfig
    
    def __init__(self, config):
        super(LlavaLlamaForCausalLMDeepQuery, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        # self.pretrain_deepquery = False
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()
        
    def initialize_lmhead(self, model_args, fsdp=None):
        pretrain_lmhead_path = getattr(model_args, 'pretrain_lmhead_path', None)
        if pretrain_lmhead_path is not None:
            query_weights = torch.load(pretrain_lmhead_path)
            print(f"loading lmhead weight from {pretrain_lmhead_path}")
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
            def get_w_half(weights, keyword):
                return {k.split(keyword + '.')[1]: v.to(torch.float16) for k, v in weights.items() if keyword in k}
            self.lm_head.half()
            self.lm_head.load_state_dict(get_w_half(query_weights, 'lm_head'))
        
    def initialize_deepquery(self, model_args, fsdp=None):
        pretrain_deepquery_path = getattr(model_args, 'pretrain_deepquery_path', None)
        num_queries = getattr(model_args, 'num_queries', 20)
        n_embd = getattr(model_args, 'n_embd', 128)
        n_layer = getattr(model_args, 'n_layer', 32)
        n_head = getattr(model_args, 'n_head', 32)
        self.deepquery = DeepQuery(seq_len=num_queries, n_embd=n_embd, n_layer=n_layer, n_head=n_head).half().to(self.device)
        self.pretrain_deepquery = True
        if pretrain_deepquery_path is not None:
            query_weights = torch.load(pretrain_deepquery_path)
            print(f"loading deepquery weight from {pretrain_deepquery_path}")
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
            def get_w_half(weights, keyword):
                return {k.split(keyword + '.')[1]: v.to(torch.float16) for k, v in weights.items() if keyword in k}
            self.deepquery.half()
            self.deepquery.load_state_dict(get_w_half(query_weights, 'deepquery'))
            # self.deepquery = self.deepquery.to(self.device)
        
    def initialize_icondeepquery(self, model_args, fsdp=None):
        pretrain_icondeepquery_path = getattr(model_args, 'pretrain_icondeepquery_path', None)
        num_imgtoken = getattr(model_args, 'num_imgtoken', 576)
        num_queries = getattr(model_args, 'num_queries', 20)
        n_embd = getattr(model_args, 'n_embd', 128)
        n_layer = getattr(model_args, 'n_layer', 32)
        n_head = getattr(model_args, 'n_head', 32)
        self.icondeepquery = IConDeepQuery(n_imgtokens=num_imgtoken, seq_len=num_queries, n_embd=n_embd, n_layer=n_layer, n_head=n_head).half().to(self.device)
        self.pretrain_icondeepquery = True
        if pretrain_icondeepquery_path is not None:
            query_weights = torch.load(pretrain_icondeepquery_path)
            print(f"loading icondeepquery weight from {pretrain_icondeepquery_path}")
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}
            def get_w_half(weights, keyword):
                return {k.split(keyword + '.')[1]: v.to(torch.float16) for k, v in weights.items() if keyword in k}
            self.icondeepquery.half()
            self.icondeepquery.load_state_dict(get_w_half(query_weights, 'icondeepquery'))
            # self.icondeepquery = self.icondeepquery.to(self.device)
        
    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, images) #TODO the interface of tokens
        # add queries to the input embedding

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        try: 
            if self.pretrain_deepquery and (past_key_values is None):
            # if False:
                bsz = images.shape[0]
                past_key_values_prompts = self.deepquery(bsz)
                past_key_values = past_key_values_prompts
                prefix_attention_mask = torch.ones(bsz, self.deepquery.seq_len).to(attention_mask.device)
                attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
        except:
            pass
        
        # IConDeepQuery
        try: 
            if self.pretrain_icondeepquery and (past_key_values is None):
                bsz = images.shape[0]
                past_key_values = self.icondeepquery.past_key_values
                prefix_attention_mask = torch.ones(bsz, self.icondeepquery.seq_len).to(attention_mask.device)
                attention_mask = torch.cat((prefix_attention_mask, attention_mask), dim=1)
                # print(attention_mask)
        except:
            pass
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache, 
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        hidden_states = outputs[0] # hidden_states: (8,886,4096)
        # print(hidden_states.shape)
        logits = self.lm_head(hidden_states) # logits: (8, 886, 32000), labels: (8, 886)
        # print(logits.shape)
        loss = None # 
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size) # vocab_size: 32000, shift_logits: (7080, 32000)
            shift_labels = shift_labels.view(-1) # 6920
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device) # 

            loss = loss_fct(shift_logits, shift_labels)

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values: # if past_key_values exist, input_ids will keep the last one
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
            }
        )
        return model_inputs

AutoConfig.register("llava", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLMDeepQuery)
