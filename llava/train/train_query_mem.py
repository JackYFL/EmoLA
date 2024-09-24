# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
# Make it more memory efficient by monkey patching the LLaMA model with FlashAttn.

# Need to call this before importing transformers.
from llava.train.llava_query_flash_atten_monkey_patch import new_replace_llama_attn_with_flash_attn
from llava.train.hack_llama import place_modeling_llama_train_forward

place_modeling_llama_train_forward()
new_replace_llama_attn_with_flash_attn()

from llava.train.train_query import train_query

if __name__ == "__main__":
    train_query()
