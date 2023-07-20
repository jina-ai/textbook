from typing import cast

import torch
from transformers import (
    GPTBigCodeConfig,
    GPTBigCodeForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
)


class StarCoderBase(GPTBigCodeForCausalLM):
    config: GPTBigCodeConfig

    def __init__(self):
        super().__init__(self.config)

    @staticmethod
    def get_tokenizer() -> PreTrainedTokenizer:
        return cast(
            PreTrainedTokenizer, AutoTokenizer.from_pretrained("bigcode/starcoder")
        )


class StarCoderTest(StarCoderBase):
    config = GPTBigCodeConfig(
        n_layer=1,
        activation_function="gelu",
        n_head=12,
        n_embd=768,
        torch_dtype=torch.bfloat16,
    )


class StarCoderTiny(StarCoderBase):
    config = GPTBigCodeConfig(
        n_layer=5,
        activation_function="gelu",
        n_head=12,
        n_embd=768,
        torch_dtype=torch.bfloat16,
    )
