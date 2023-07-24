import torch
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    AutoConfig,
    PreTrainedModel,
    AutoModelForCausalLM,
)


class ReplitBase:
    tokenizer: PreTrainedTokenizer
    model: PreTrainedModel
    base_model = "replit/replit-code-v1-3b"

    config = AutoConfig.from_pretrained(
        "replit/replit-code-v1-3b",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        init_device="cuda",
    )

    def __init__(self):
        self._init_tokenizer()
        self.model = AutoModelForCausalLM.from_pretrained(
            self.base_model, config=self.config, trust_remote_code=True
        )

    def _init_tokenizer(self):
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.base_model, trust_remote_code=True
        )
        self.tokenizer.padding_side = "left"  # Allow batched inference
        self.tokenizer.pad_token = self.tokenizer.eos_token


class ReplitDebug(ReplitBase):
    config = AutoConfig.from_pretrained(
        "replit/replit-code-v1-3b",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        init_device="cuda",
        n_layers=1,
    )
